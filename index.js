var http = require('http');

http.createServer(function (req, res) {
  res.writeHead(200, {'Content-Type': 'text/html'});
  model.detect('./indoor9.mp4').then(predictions => { 
        res.end("yay")
});
}).listen(3000);
console.log("Listening on port 3000");