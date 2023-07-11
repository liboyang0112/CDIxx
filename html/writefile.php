<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
  move_uploaded_file('php://input', "cache.txt");
  file_put_contents(substr($_SERVER['PATH_INFO'], 1), file_get_contents('php://input'));
  echo "saved to file: ".substr($_SERVER['PATH_INFO'], 1);
}
?>
