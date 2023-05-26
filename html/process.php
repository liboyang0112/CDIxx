<?php 
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
  if (isset($_FILES['files'])) {
    $path = $_POST['path'];
    $all_files = count($_FILES['files']['tmp_name']);
    for ($i = 0; $i < $all_files; $i++) {  
      $file_name = $_FILES['files']['name'][$i];
      $file_tmp = $_FILES['files']['tmp_name'][$i];
      $file = $path . "/" . $file_name;
      echo $file;
      move_uploaded_file($file_tmp, $file);
    }
  }
}
