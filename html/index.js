const http = require("http"),
  fs = require("fs")
var server = http.createServer(function(request,response){
  dat = '';
  response.setHeader("Access-Control-Allow-Origin","*");
  console.log("received request: "+ request.url);
  request.on('error', (err) => {
    console.error(err.stack);
  }).on('data', data=>{
    dat+=data;
    response.statusCode = 200;
    response.on('error', err=>{console.error(err.stack);});
    if(request.url == "/read"){
      if(fs.existsSync (dat))
        response.write(fs.readFileSync(dat,"utf8"));
    }else if(request.url.startsWith("/save/")){
      const file = request.url.replace("/save/","");
      fs.writeFile(file,dat,err => {if (err) throw err; console.log('Saved to '+file + ' !');});
      response.write("saved");
    }else{
      response.write("URL not accessible!");
    }
    response.end();
    return;
  });
  return;
}).listen(8080);
console.log('Server is running at http://localhost:8080/');
