<?php
// turn on errors while you’re testing
ini_set('display_errors', 1);
error_reporting(E_ALL);

require __DIR__ . '/vendor/autoload.php';
use GeoIp2\Database\Reader;

// 1) GeoIP lookup
$reader = new Reader(__DIR__ . '/GeoLite2-Country.mmdb');
$ip = ($_SERVER['REMOTE_ADDR'] === '::1') ? '127.0.0.1' : $_SERVER['REMOTE_ADDR'];
try {
    $rec         = $reader->country($ip);
    $countryCode = $rec->country->isoCode  ?? 'Unknown';
    $countryName = $rec->country->name     ?? 'Unknown';
} catch (\Exception $e) {
    $countryCode = $countryName = 'Unknown';
}

// 2) Log to Strato’s MySQL
try {
    $pdo = new PDO(
        'mysql:host=database-5017920693.webspace-host.com;dbname=dbs14266576;charset=utf8mb4',
        'dbu190527',
        'QuantumCommSys2024!',
        [
          PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
          PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        ]
    );
    $stmt = $pdo->prepare(
      'INSERT INTO visits (ip, country_code, country_name, visited_at)
       VALUES (?, ?, ?, NOW())'
    );
    $stmt->execute([$ip, $countryCode, $countryName]);
} catch (\Exception $e) {
    // don’t expose errors to users in production
    error_log("Tracker error: " . $e->getMessage());
}

// 3) Return a 1×1 transparent GIF
header('Content-Type: image/gif');
echo base64_decode(
  'R0lGODlhAQABAPAAAP///wAAACwAAAAAAQABAEACAkQBADs='
);
