var TerrainGenerator = function(canvas_width, canvas_height, grid_width, grid_height, context, terrain_map_name) {
  const tileSize = 16;
  const renderTileSize = canvas_width / grid_width;
  const rowTileCount = grid_height;
  const colTileCount = grid_width;
  const imageNumTiles = 27;
  current_terrain_map_name = terrain_map_name;
  let map = TileMaps[current_terrain_map_name];

  let tileset = new Image();
  tileset.src = 'local/images/tileset.png';

  this.drawTerrain = function() {
    for (let layer = 0; layer < map.layers.length; layer++) {
      for (let r = 0; r < rowTileCount; r++) {
        for (let c = 0; c < colTileCount; c++) {
          var tile = map.layers[layer].data[ (r*colTileCount) + c] - 1;
          var tileRow = (tile / imageNumTiles) | 0; // Bitwise OR operation
          var tileCol = (tile % imageNumTiles) | 0;

          context.drawImage(tileset, (tileCol * tileSize), (tileRow * tileSize), tileSize, tileSize, (c * renderTileSize), (r * renderTileSize), renderTileSize, renderTileSize);
        }
      }
    }
  }

  this.resetCanvas = function() {
    this.drawTerrain();
  }

  ws.addEventListener('message', function(message) {
    const msg = JSON.parse(message.data);
    switch (msg["type"]) {
      case "terrain_map_name":
        map = TileMaps[msg['value']]
        this.drawTerrain()
        break;
    }
  }.bind(this))
};
