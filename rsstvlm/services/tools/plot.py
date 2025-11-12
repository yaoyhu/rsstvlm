import json

import h5py
import numpy as np
from llama_index.core.llms import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)

from rsstvlm.utils import qwen3_vl_30b


class H5Plot:
    NAME = "H5Plot"
    DESCRIPTION = "Plot images for H5 file(s)."

    def structure(self, h5_path: str):
        """
        Returns the hierarchical structure of an H5 file, showing all groups
        and datasets with their shapes and data types.

        Args:
            h5_path: Path to the HDF5 file to inspect

        Returns:
            json: A formatted json representing the structure of the H5 file
        """

        def build_structure(name, obj):
            result = {}
            if isinstance(obj, h5py.Dataset):
                result = {
                    "type": "Dataset",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }
            elif isinstance(obj, h5py.Group):
                result = {"type": "Group"}
            structure[name] = result

        structure = {}
        with h5py.File(h5_path, "r") as f:
            f.visititems(build_structure)

        return json.dumps(structure, indent=2)

    def plot(
        self,
        h5_path: str,
        dataset_path: str = "Data/CloudFraction",
        output_path: str = "/exports/yaoyhu/rsstvlm/tests/plot.png",
    ):
        """
        Plots a specific dataset from an H5 file.
        Supports both grayscale (2D) and RGB (3D) images.

        Args:
            h5_path: Path to the HDF5 file containing image data.
            dataset_path: The full path to the dataset within the H5 file (default: "Data/CloudFraction").
            output_path: Path where the plot image will be saved (default: "/exports/yaoyhu/rsstvlm/tests/plot.png")

        Returns:
            str: Path to the saved image file, or None if the dataset is not found.

        Saves:
            A matplotlib figure showing the specified dataset,
            with a colorbar and the dataset key as title.
        """
        import matplotlib.pyplot as plt

        try:
            with h5py.File(h5_path, "r") as f:
                data = f[dataset_path]
                if not (
                    isinstance(data, h5py.Dataset) and len(data.shape) >= 2
                ):
                    print(
                        f"Path '{dataset_path}' is not a plottable dataset (must be >= 2D)."
                    )
                    return None

                img = np.array(data)

                plt.figure(figsize=(10, 8))
                if len(img.shape) == 2:  # Grayscale
                    plt.imshow(
                        img, cmap="viridis"
                    )  # Use a more scientific colormap
                elif len(img.shape) == 3:  # RGB or similar
                    # Assuming the channel is the last dimension
                    plt.imshow(img)

                plt.title(f"Dataset: {dataset_path}")
                plt.colorbar(label="Value")
                plt.xlabel("X-axis")
                plt.ylabel("Y-axis")
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"Image saved to {output_path}")
                return output_path
        except Exception as e:
            print(f"An error occurred while plotting {h5_path}: {e}")
            return None

    def visual_explain(self, image_path: str, query: str) -> str:
        """
        Uses a vision-language model to generate an detailed explanation for the given image.

        Args:
            image_path: Path to the image file to be explained.
            query: The question or prompt to guide the explanation.

        Returns:
            str: The generated explanation text.
        """
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    TextBlock(text=query),
                    ImageBlock(path=image_path),
                    # ImageBlock(url="https://example.com/image.jpg")
                ],
            )
        ]

        response = qwen3_vl_30b.chat(
            messages,
            temperature=0,
            max_tokens=1024,
        )
        return response.raw.choices[0].message.content


if __name__ == "__main__":
    h5plot = H5Plot()
    output_path = "/exports/yaoyhu/rsstvlm/tests/plot.png"

    # Visual explain
    query = "Describe the content of this image in detail."
    explanation = h5plot.visual_explain(output_path, query)
    print(explanation)
