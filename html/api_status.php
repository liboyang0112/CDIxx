<?php
header('Content-Type: application/json');
header('Cache-Control: no-cache, no-store, must-revalidate');

$workdir = $_GET['workdir'] ?? '';
$type = $_GET['type'] ?? 'ssptycho';

if (!$workdir || !$type) {
    http_response_code(400);
    echo json_encode(['error' => 'Missing workdir or type']);
    exit;
}

if (!preg_match('/^[\w.\-\/]+$/', $workdir) || !preg_match('/^\w+$/', $type)) {
    http_response_code(400);
    echo json_encode(['error' => 'Invalid parameters']);
    exit;
}

$map = [
    'ssptycho'    => ['prefix' => '.ssptycho_run_',    'log' => 'ssptycho.log'],
    'ptycho'      => ['prefix' => '.ptycho_run_',      'log' => 'ptycho.log'],
    'multiptycho' => ['prefix' => '.multiptycho_run_', 'log' => 'multiptycho.log'],
    'cdi'         => ['prefix' => '.cdi_run_',         'log' => 'cdi.log'],
];

if (!isset($map[$type])) {
    http_response_code(400);
    echo json_encode(['error' => 'Invalid type']);
    exit;
}

$prefix = $map[$type]['prefix'];
$logFile = $map[$type]['log'];

// Find latest tracking file (no shell)
$files = glob($workdir . '/' . $prefix . '*') ?: [];
if (empty($files)) {
    echo json_encode(['running' => false, 'pid' => null, 'log' => '']);
    exit;
}
usort($files, function($a, $b) {
    $ma = @filemtime($a);
    $mb = @filemtime($b);
    return ($mb === false ? 0 : $mb) - ($ma === false ? 0 : $ma);
});
$trackingFile = $files[0];

// Read PID
$pid = trim((string)@file_get_contents($trackingFile));
if (!$pid || !ctype_digit($pid)) {
    echo json_encode(['running' => false, 'pid' => null, 'log' => '']);
    exit;
}
$pid = (int)$pid;

// Check process liveness via /proc (Linux, no shell)
$running = is_dir("/proc/$pid");

// Read full log (no shell)
$logContent = '';
$logPath = $workdir . '/' . $logFile;
if (is_file($logPath)) {
    $logContent = (string)@file_get_contents($logPath);
}

echo json_encode(['running' => $running, 'pid' => $pid, 'log' => $logContent]);
?>
