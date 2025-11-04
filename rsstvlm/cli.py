import argparse

from rsstvlm.agent.agentic_rag import AgenticRAG

__project__: str = "rsstvlm"
__version__: str = "0.0.1"


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

    assert options.query is not None, "The --query argument is required."

    agent = await AgenticRAG.create()
    await agent.stream(options.query)
