<?php 
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
  $path = file_get_contents('php://input');
  if(!file_exists($path)){
    shell_exec("./createWorkDir.sh " . $path);
    echo "create";
  }else{
    echo "exists";
  };
}
