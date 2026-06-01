import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATConv, Sequential, GraphNorm,
    global_mean_pool, global_max_pool,
)
from transformers import AutoModelForMaskedLM
from src.branch_3D.model_unimol import UniMolModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2D Branch
# =============================================================================

class GNN_branch_with_GAT(torch.nn.Module):
    def __init__(self):
        super(GNN_branch_with_GAT, self).__init__()
        self.graphconv = Sequential('x,edge_index,batch', [
            (GATConv(9, 128),    'x,edge_index -> x'),
            nn.LeakyReLU(),
            GraphNorm(128),
            (nn.Dropout(p=0.1), 'x -> x'),
            (GATConv(128, 256),  'x,edge_index -> x'),
            nn.LeakyReLU(),
            GraphNorm(256),
            (nn.Dropout(p=0.1), 'x -> x'),
            (GATConv(256, 512),  'x,edge_index -> x'),
            nn.LeakyReLU(),
            GraphNorm(512),
            (nn.Dropout(p=0.1), 'x -> x'),
            (GATConv(512, 767),  'x,edge_index -> x'),
            nn.LeakyReLU(),
            GraphNorm(767),
        ])

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        graph_representation = self.graphconv(x, edge_index, batch)

        mean = global_mean_pool(graph_representation, batch).view(-1, 1, 767)
        max_ = global_max_pool(graph_representation, batch).view(-1, 1, 767)

        # [B, 2, 767]
        return torch.cat([mean, max_], dim=1)


# =============================================================================
# Main Model
# =============================================================================

class Net(torch.nn.Module):
    """
    Three-modality cross-attention model.

    Forward pass (shared):
        seq_2d [B, 2, 767] --query--> DecoderLayer1 <--memory-- seq_1d [B, T, 767]
                                            |
                                      att_2d_1d [B, 2, 767]

        seq_2d [B, 2, 767] --query--> DecoderLayer2 <--memory-- seq_3d [B, N, 767]
                                            |
                                      att_2d_3d [B, 2, 767]

    Fusion options (controlled by self.fusion):

        fusion="concat":
            cat(att_2d_1d, att_2d_3d, dim=1) --> [B, 4, 767]
            flatten                           --> [B, 3068]
            concat_head                       --> [B, n_tasks]

        fusion="plus":
            att_2d_1d + att_2d_3d            --> [B, 2, 767]
            flatten                           --> [B, 1534]
            plus_head                         --> [B, n_tasks]

    Training stages:
        "fusion_only" : freeze 1D (ChemBERTa) + 3D (UniMol)
                        train  2D (GAT) + fusion layers + heads
        "unfreeze_all": train everything
    """

    VALID_FUSIONS = {"concat", "plus"}

    def __init__(self, n_output_layers=1, fusion="concat"):
        super().__init__()

        if fusion not in self.VALID_FUSIONS:
            raise ValueError(
                f"Unknown fusion: '{fusion}'. "
                f"Valid options: {self.VALID_FUSIONS}"
            )

        self.hidden_dim      = 767
        self.n_output_layers = n_output_layers
        self.fusion          = fusion

        # ------------------------------------------------------------------ #
        # Branches                                                             #
        # ------------------------------------------------------------------ #
        self.gnn_branch = GNN_branch_with_GAT()

        self.geometric_branch = UniMolModel(
            output_dim=self.hidden_dim,
            data_type="molecule",
            remove_hs=False,
        )

        self.model = AutoModelForMaskedLM.from_pretrained(
            "seyonec/ChemBERTa-zinc-base-v1"
        )

        # ------------------------------------------------------------------ #
        # Cross-attention layers                                               #
        # ------------------------------------------------------------------ #
        self.decodelayer1 = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, nhead=13, batch_first=True,
        )
        self.decodelayer2 = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, nhead=13, batch_first=True,
        )

        # ------------------------------------------------------------------ #
        # Concat head                                                          #
        # cat(att_2d_1d, att_2d_3d, dim=1) -> [B, 4, 767]                   #
        # flatten                           -> [B, 4 * 767] = [B, 3068]      #
        # ------------------------------------------------------------------ #
        self.concat_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 20),
            nn.LeakyReLU(),
            nn.Linear(20, self.n_output_layers),
        )

        # ------------------------------------------------------------------ #
        # Plus head                                                            #
        # att_2d_1d + att_2d_3d -> [B, 2, 767]                               #
        # flatten               -> [B, 2 * 767] = [B, 1534]                  #
        # ------------------------------------------------------------------ #
        self.plus_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 20),
            nn.LeakyReLU(),
            nn.Linear(20, self.n_output_layers),
        )

    # ---------------------------------------------------------------------- #
    # Staged training                                                          #
    # ---------------------------------------------------------------------- #

    def set_train_stage(self, stage: str):
        def freeze(m: nn.Module):
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

        def unfreeze(m: nn.Module):
            for p in m.parameters():
                p.requires_grad = True
            m.train()

        fusion_modules = [
            self.decodelayer1,
            self.decodelayer2,
            self.concat_head,
            self.plus_head,
        ]

        if stage == "fusion_only":
            freeze(self.model)
            freeze(self.geometric_branch)
            unfreeze(self.gnn_branch)
            for m in fusion_modules:
                unfreeze(m)

        elif stage == "unfreeze_all":
            unfreeze(self.model)
            unfreeze(self.geometric_branch)
            unfreeze(self.gnn_branch)
            for m in fusion_modules:
                unfreeze(m)

        else:
            raise ValueError(f"Unknown stage: '{stage}'")

    # ---------------------------------------------------------------------- #
    # Branch helpers                                                           #
    # ---------------------------------------------------------------------- #

    def _get_1d_representation(self, inputs):
        """
        Returns
        -------
        seq_1d : [B, T, 767]
        """
        if not any(p.requires_grad for p in self.model.parameters()):
            with torch.no_grad():
                out = self.model(**inputs)
        else:
            out = self.model(**inputs)

        return out[0]   # [B, T, 767]

    def _get_2d_representation(self, graph):
        """
        Returns
        -------
        seq_2d : [B, 2, 767]
        """
        return self.gnn_branch(graph)

    def _get_3d_representation(self, unimol_input):
        """
        Returns
        -------
        seq_3d : [B, N, 767]
        """
        if not any(p.requires_grad for p in self.geometric_branch.parameters()):
            with torch.no_grad():
                geo_out = self.geometric_branch(
                    unimol_input["src_tokens"],
                    unimol_input["src_distance"],
                    unimol_input["src_coord"],
                    unimol_input["src_edge_type"],
                )
        else:
            geo_out = self.geometric_branch(
                unimol_input["src_tokens"],
                unimol_input["src_distance"],
                unimol_input["src_coord"],
                unimol_input["src_edge_type"],
            )

        if geo_out.dim() == 2:
            return geo_out.unsqueeze(1)   # [B, 1, 767]
        return geo_out                    # [B, N, 767]

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(self, graph, inputs, unimol_input):
        # ── Encode ────────────────────────────────────────────────────────
        seq_1d = self._get_1d_representation(inputs)       # [B, T, 767]
        seq_2d = self._get_2d_representation(graph)        # [B, 2, 767]
        seq_3d = self._get_3d_representation(unimol_input) # [B, N, 767]

        # ── Cross-attention: 2D queries 1D ────────────────────────────────
        att_2d_1d = self.decodelayer1(
            seq_2d,   # query  [B, 2, 767]
            seq_1d,   # memory [B, T, 767]
        )             # output [B, 2, 767]

        # ── Cross-attention: 2D queries 3D ────────────────────────────────
        att_2d_3d = self.decodelayer2(
            seq_2d,   # query  [B, 2, 767]
            seq_3d,   # memory [B, N, 767]
        )             # output [B, 2, 767]

        # ── Fusion ────────────────────────────────────────────────────────
        if self.fusion == "concat":
            # [B, 2, 767] cat [B, 2, 767] → [B, 4, 767] → [B, 3068]
            fused = torch.cat([att_2d_1d, att_2d_3d], dim=1)
            fused = fused.view(fused.size(0), -1)
            return self.concat_head(fused)             # [B, n_tasks]

        if self.fusion == "plus":
            # [B, 2, 767] + [B, 2, 767] → [B, 2, 767] → [B, 1534]
            fused = att_2d_1d + att_2d_3d
            fused = fused.view(fused.size(0), -1)
            return self.plus_head(fused)               # [B, n_tasks]