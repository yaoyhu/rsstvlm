import argparse

from rsstvlm.agent.agentic_rag import AgenticRAG

__project__: str = "rsstvlm"
__version__: str = "0.0.2"


async def main():
    parser = argparse.ArgumentParser(
        description=f"CLI for {__project__}:{__version__}"
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"{__project__} {__version__}",
        help="Show program's version number and exit.",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        dest="query",
        help="The query to be processed.",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        dest="path",
        help="The path to your data files.",
    )

    options = parser.parse_args()

    # assert options.query is not None, "The --query argument is required."
    querys = [
        # "Given graph database, What differs from Aligner and RLHF?",
        "Given H5 file path /satellite/EMI_NO2/EMI_NO2_LV2_v2.0/GF5_EMI_20190608_005765_L10000007826_VI1.h5, follow these steps: Step 1: Peek the structure of the h5 file to understand its contents. Step 2: Plot the dataset and save it, then return the output image path and stop. Step 3: Use visual_explain to analyze and describe the image content in detail. Stop after completing all steps.",
        # "Analyze the image and give a detailed description combine with database knowledge of CloudFraction.",
    ]

    agent = await AgenticRAG.create()
    await agent.stream(querys[0])
