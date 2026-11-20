<?php
// Pass-through proxy: connects to Unix socket and streams raw data to browser
// The C++ side already sends complete HTTP + multipart MJPEG
// Usage: /live.php?sock=/full/path/to/cdixx_stream.sock

// Tell nginx NOT to buffer this response
header('X-Accel-Buffering: no');
header('Content-Type: multipart/x-mixed-replace; boundary=frame');
header('Cache-Control: no-cache, no-store');

// Disable ALL PHP output buffering
while (ob_get_level()) ob_end_clean();
ob_implicit_flush(true);

$sockPath = isset($_GET['sock']) ? $_GET['sock'] : '';
if (empty($sockPath) || !preg_match('/^[\w.\-\/]+$/', $sockPath)) {
    http_response_code(400);
    header('Content-Type: text/plain');
    echo 'Invalid sock parameter';
    exit;
}

if ($sockPath[0] !== '/') {
    $sockPath = __DIR__ . '/' . $sockPath;
}

// If absolute path doesn't exist, try relative to web root (symlink)
if (!file_exists($sockPath)) {
    $altPath = __DIR__ . $sockPath;
    if (file_exists($altPath)) {
        $sockPath = $altPath;
    } else {
        http_response_code(502);
        header('Content-Type: text/plain');
        echo 'Socket not found: ' . $sockPath;
        exit;
    }
}

$fp = @stream_socket_client('unix://' . $sockPath, $errno, $errstr, 5);
if (!$fp) {
    http_response_code(502);
    header('Content-Type: text/plain');
    echo 'Connect failed: ' . $errstr;
    exit;
}

stream_set_blocking($fp, false);

// Pure pass-through: read chunks, echo immediately
while (true) {
    if (connection_aborted()) break;
    $data = fread($fp, 65536);
    if ($data === '' || $data === false) {
        if (feof($fp)) break;
        usleep(1000);
        continue;
    }
    echo $data;
    flush();
}

@fclose($fp);
