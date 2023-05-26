<?php 

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
  if (isset($_FILES['files'])) {
    $errors = [];
    $path = $_POST['path'];
    if(!file_exists($path)){
      shell_exec("./createWorkDir.sh " . $path);
    };
    $extensions = ['jpg', 'jpeg', 'png', 'gif'];
    $all_files = count($_FILES['files']['tmp_name']);


    for ($i = 0; $i < $all_files; $i++) {  
      $file_name = $_FILES['files']['name'][$i];
      $file_tmp = $_FILES['files']['tmp_name'][$i];
      $file_type = $_FILES['files']['type'][$i];
      $file_size = $_FILES['files']['size'][$i];
      $tmp = explode('.', $_FILES['files']['name'][$i]);
      $file_ext = strtolower(end($tmp));

      $file = $path . "/" . $file_name;
      echo $file;
      move_uploaded_file($file_tmp, $file);
    }
  }
}
