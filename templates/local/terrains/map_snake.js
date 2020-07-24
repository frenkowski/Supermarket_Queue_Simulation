(function(name,data){
 if(typeof onTileMapLoaded === 'undefined') {
  if(typeof TileMaps === 'undefined') TileMaps = {};
  TileMaps[name] = data;
 } else {
  onTileMapLoaded(name,data);
 }
 if(typeof module === 'object' && module && module.exports) {
  module.exports = data;
 }})("map_snake",
{ "compressionlevel":-1,
 "editorsettings":
    {
     "export":
        {
         "target":"..\/..\/..\/resources"
        }
    },
 "height":28,
 "infinite":false,
 "layers":[
        {
         "data":[334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 334, 334, 334, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 334, 334, 1055, 1056, 1056, 1056, 1030, 1055, 1027, 1056, 1030, 1055, 1055, 1055, 1027, 1030, 1055, 1055, 1055, 1027, 1056, 1056, 1056, 1056, 1056, 1056, 1056, 1056, 1056, 1056, 1056, 1056, 334, 334, 1055, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 334, 334, 1055, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 334, 334, 1055, 1054, 1054, 1054, 1054, 1054, 1028, 1054, 1054, 1054, 1054, 1028, 1054, 1054, 1054, 1054, 1028, 1054, 1054, 1054, 1054, 1028, 1054, 1054, 1054, 1054, 1028, 1054, 1054, 1054, 334, 334, 1055, 1028, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 334, 334, 1055, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 334, 334, 1055, 1058, 1054, 1054, 1029, 1056, 1057, 1054, 1054, 1029, 1056, 1057, 1054, 1054, 1029, 1056, 1057, 1054, 1054, 1029, 1056, 1057, 1054, 1054, 1029, 1056, 1057, 1054, 1054, 1054, 334, 334, 1055, 1058, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 334, 334, 1055, 1058, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 334, 334, 1055, 1058, 1054, 1054, 1054, 1054, 1028, 1054, 1054, 1054, 1054, 1028, 1054, 1054, 1054, 1054, 1028, 1054, 1054, 1054, 1054, 1028, 1054, 1054, 1054, 1054, 1028, 1054, 1054, 1054, 334, 334, 1055, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 334, 334, 1055, 1057, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 334, 334, 1055, 1054, 1054, 1054, 1029, 1056, 1057, 1054, 1054, 1029, 1056, 1057, 1054, 1054, 1029, 1056, 1057, 1054, 1054, 1029, 1056, 1057, 1054, 1054, 1029, 1056, 1057, 1054, 1054, 1054, 334, 334, 1055, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 334, 334, 1055, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 334, 334, 1055, 1054, 1054, 1054, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1028, 334, 334, 1055, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1058, 334, 334, 1055, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1064, 1054, 1054, 1054, 1058, 334, 334, 1055, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 334, 334, 1055, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 334, 334, 1055, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 1054, 334, 334, 1055, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1054, 334, 334, 1055, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1054, 334, 334, 1055, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1058, 1054, 1054, 1054, 1054, 1054, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334],
         "height":28,
         "id":1,
         "name":"Floor\/Void",
         "opacity":1,
         "type":"tilelayer",
         "visible":true,
         "width":32,
         "x":0,
         "y":0
        }, 
        {
         "data":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1009, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1011, 0, 0, 1036, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1037, 1038, 0, 0, 1063, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1065, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 367, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 366, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         "height":28,
         "id":3,
         "name":"Walls\/Corner Clip",
         "opacity":1,
         "type":"tilelayer",
         "visible":true,
         "width":32,
         "x":0,
         "y":0
        }, 
        {
         "data":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 412, 412, 0, 0, 1012, 1013, 1012, 1013, 0, 1012, 1013, 1012, 1013, 0, 0, 1004, 1004, 0, 0, 0, 1004, 1004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 439, 439, 0, 0, 1039, 1040, 1039, 1040, 0, 1039, 1040, 1039, 1040, 0, 0, 1031, 1031, 0, 0, 0, 1031, 1031, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 466, 466, 0, 0, 1066, 1067, 1066, 1067, 0, 1066, 1067, 1066, 1067, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 931, 932, 0, 0, 0, 931, 932, 0, 0, 0, 931, 932, 0, 0, 0, 931, 932, 0, 0, 0, 931, 932, 0, 0, 0, 0, 0, 0, 932, 0, 0, 0, 958, 959, 0, 0, 0, 958, 959, 0, 0, 0, 958, 959, 0, 0, 0, 958, 959, 0, 0, 0, 958, 959, 0, 0, 0, 0, 0, 0, 959, 0, 0, 0, 985, 986, 0, 0, 0, 985, 986, 0, 0, 0, 985, 986, 0, 0, 0, 985, 986, 0, 0, 0, 985, 986, 0, 0, 0, 0, 0, 0, 932, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1062, 0, 0, 959, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1041, 0, 0, 932, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1035, 0, 0, 959, 0, 0, 0, 931, 932, 0, 0, 0, 931, 932, 0, 0, 0, 931, 932, 0, 0, 0, 931, 932, 0, 0, 0, 931, 932, 0, 0, 0, 0, 0, 0, 986, 0, 0, 0, 958, 959, 0, 0, 0, 958, 959, 0, 0, 0, 958, 959, 0, 0, 0, 958, 959, 0, 0, 0, 958, 959, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 985, 986, 0, 0, 0, 985, 986, 0, 0, 0, 985, 986, 0, 0, 0, 985, 986, 0, 0, 0, 985, 986, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1032, 0, 0, 0, 0, 1032, 0, 0, 0, 0, 1032, 0, 0, 0, 0, 1032, 0, 0, 0, 0, 1032, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1059, 0, 0, 0, 0, 1059, 0, 0, 0, 0, 1059, 0, 0, 0, 0, 1059, 0, 0, 0, 0, 1059, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1005, 871, 0, 0, 0, 1005, 871, 0, 0, 0, 1005, 871, 0, 0, 0, 1005, 871, 0, 0, 0, 1005, 871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1060, 0, 0, 0, 0, 1060, 0, 0, 0, 0, 1060, 0, 0, 0, 0, 1060, 0, 0, 0, 0, 1060, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         "height":28,
         "id":4,
         "name":"Objects",
         "opacity":1,
         "type":"tilelayer",
         "visible":true,
         "width":32,
         "x":0,
         "y":0
        }, 
        {
         "data":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 957, 0, 0, 0, 952, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 977, 0, 0, 975, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 979, 0, 0, 926, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 1001, 927, 0, 0, 0, 1006, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         "height":28,
         "id":6,
         "name":"Snake",
         "opacity":1,
         "type":"tilelayer",
         "visible":true,
         "width":32,
         "x":0,
         "y":0
        }],
 "nextlayerid":8,
 "nextobjectid":1,
 "orientation":"orthogonal",
 "properties":[
        {
         "name":"capacity",
         "type":"int",
         "value":25
        }, 
        {
         "name":"lane_switch_boundary",
         "type":"int",
         "value":6
        }],
 "renderorder":"right-down",
 "tiledversion":"1.4.1",
 "tileheight":16,
 "tilesets":[
        {
         "firstgid":1,
         "source":"..\/..\/..\/..\/Supermarket.tsx"
        }],
 "tilewidth":16,
 "type":"map",
 "version":1.4,
 "width":32
});