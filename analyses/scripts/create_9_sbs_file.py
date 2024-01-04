from pathlib import Path
import re
import pandas as pd
from GenomeSigInfer.sbs import SBSMatrixGenerator

# _PATH = Path(__file__).parent.parent.parent / "data" / "vcf"
# files = (_PATH / "WES_Other.20180327.simple", _PATH / "WGS_Other.20180413.simple")
# sbs_9_context = SBSMatrixGenerator.generate_single_sbs_matrix("./analyses/SBS/", files, "./analyses/ref_genome/GRCh37", "GRCh37", max_context=9)
# filename = Path(__file__).parent.parent / "SBS" / f"sbs.{sbs_9_context.shape[0]}.parquet"
filename = Path(__file__).parent.parent / "SBS" / f"sbs.393216.parquet"
filename = Path(__file__).parent.parent / "SBS" / "s.parquet"
sbs_9_context = pd.read_parquet(filename)
print(sbs_9_context)
# sbs_9_context.to_parquet(filename)

# NUCL_MAP = {
#     "nucl_strength": {
#         "W": ["A", "T"], "S": ["C", "G"]
#     },
#     "amino": {
#         "M": ["A", "C"], "K": ["G", "T"]
#     },
#     "structure": {
#         "R": ["A", "G"], "Y": ["C", "T"]
#     }
# }

NUCL_MAP = {
    "nucl_strength": {"A": "W", "T": "W", "C": "S", "G": "S"},
    "amino": {"A": "M", "C": "M", "G": "K", "T": "K"},
    "structure": {"A": "R", "C": "Y", "G": "R", "T": "Y"}
}

def cluster(df, cluster_type, filename):
    df['MutationType'] = df['MutationType'].apply(lambda x: ''.join(cluster_type.get(c, c) for c in x[:2]) + x[2:-2] + ''.join(cluster_type.get(c, c) for c in x[-2:]))
    df = df.groupby("MutationType").sum()
    print(df)


cluster(sbs_9_context, NUCL_MAP["nucl_strength"])