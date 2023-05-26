const http = require("http"),
  fs = require("fs"),
  url = require("url")
var server = http.createServer(function(request,response){
  dat = '';
  const { headers, method, url } = request;
  response.setHeader("Access-Control-Allow-Origin","*");
  request.on('error', (err) => {
    console.error(err.stack);
  }).on('data', data=>{
    dat+=data;
    response.statusCode = 200;
    response.on('error', err=>{console.error(err.stack);});
    if(request.url == "/path_pulse"){
      dat='/home/boyang/html/images/'+dat+'/';
      const data = fs.readFileSync(dat+"pulse.cfg","utf8");
      response.end(data);
      return;
    }else if(request.url == "/path_cdi"){
      dat='/home/boyang/html/images/'+dat+'/';
      const data = fs.readFileSync(dat+"cdi.cfg","utf8");
      response.end(data);
      return;
    }else if(request.url.startsWith("/save/")){
      const file = request.url.replace("/save/","");
      fs.writeFile(file,dat,err => {if (err) throw err; console.log('Saved!');});
    }
  });
  return;
}).listen(8080);
console.log('Server is running at http://localhost:8080/');
