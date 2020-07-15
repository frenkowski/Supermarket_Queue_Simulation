from mesa.visualization.modules import CanvasGrid


class CanvasGridWithTerrain(CanvasGrid):
    package_includes = ["InteractionHandler.js"]
    local_includes = ["CanvasModule.js", "TerrainGenerator.js", "GridDraw.js", "terrains/map3.js", "terrains/map3-snake.js"]

    def __init__(
        self,
        portrayal_method,
        grid_width,
        grid_height,
        terrain_map_name,
        canvas_width=500,
        canvas_height=500,
    ):
        """ Instantiate a new CanvasGrid.

        Args:
            portrayal_method: function to convert each object on the grid to
                              a portrayal, as described above.
            grid_width, grid_height: Size of the grid, in cells.
            canvas_height, canvas_width: Size of the canvas to draw in the
                                         client, in pixels. (default: 500x500)

        """
        self.portrayal_method = portrayal_method
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        new_element = "new CanvasModule({}, {}, {}, {}, '{}')".format(
            self.canvas_width, self.canvas_height, self.grid_width, self.grid_height, terrain_map_name
        )

        self.js_code = "elements.push(" + new_element + ");"
