<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
  $filename = file_get_contents('php://input');
  echo file_get_contents($filename);
}
?>
