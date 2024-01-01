from pathlib import Path
from GenomeSigInfer.sbs import SBSMatrixGenerator

_PATH = Path(__file__).parent.parent.parent / "data" / "vcf"
files = (_PATH / "WES_Other.20180327.simple", _PATH / "WGS_Other.20180413.simple")
sbs_9_context = SBSMatrixGenerator.generate_single_sbs_matrix("./analyses/SBS/", files, "./analyses/ref_genome/GRCh37", "GRCh37", max_context=9)
filename = Path(__file__).parent.parent / "SBS" / f"sbs.{sbs_9_context.shape[0]}.parquet"
sbs_9_context.to_parquet(filename)

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

# def replace_nucleotides(df, cluster_type, columns):
#     """
#     Replace nucleotides in specific columns of the DataFrame.

#     Args:
#         df (pandas.DataFrame): The DataFrame to modify.
#     """
#     for col in columns:
#         for key, value in cluster_type.items():
#             df.loc[df[col].isin(value), col] = key

# def cluster(df):
#     col_index = {"-4": 0, "-3": 1, "3": 11, "4": 12}
#     for col, index in col_index.items():
#         df[col] = df["MutationType"].str[index]
#     replace_nucleotides(df, NUCL_MAP["nucl_strength"], col_index.keys())
#     df["MutationType"] = df["-4"] + df["-3"] + df["MutationType"].str[2:11] + df["3"] + df["4"]
#     df.drop(["-4", "-3", "3", "4"], axis=1, inplace=True)
#     print(df.groupby("MutationType").sum())
