class H5Plot:
    NAME = "H5Plot"
    DESCRIPTION = "Plot images for H5 file(s)."

    def ret_images(self):
        """This is a tool for plot h5 files."""
        return 1


def get_tool_class():
    return H5Plot
