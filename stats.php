<?php
// 1) Connect to Strato MySQL
$pdo = new PDO(
    'mysql:host=database-5017920693.webspace-host.com;dbname=dbs14266576;charset=utf8mb4',
    'dbu190527',
    'QuantumCommSys2024!',  // your password
    [ PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION ]
);


// 2) Fetch total visits
$total = (int)$pdo->query('SELECT COUNT(*) FROM visits')->fetchColumn();

// 3) Fetch per-country counts, sorted descending
$countries = $pdo->query(
  'SELECT country_name, country_code, COUNT(*) AS cnt
   FROM visits
   GROUP BY country_code, country_name
   ORDER BY cnt DESC'
)->fetchAll(PDO::FETCH_ASSOC);
?><!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Visitor Stats</title>
  <style>
    body { font-family: sans-serif; padding: 1rem; }
    table { border-collapse: collapse; width: 100%; max-width: 600px; }
    th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; }
    th { background: #f0f0f0; }
  </style>
</head>
<body>
  <h1>Total visits: <?= htmlspecialchars($total) ?></h1>

  <h2>By Country</h2>
  <table>
    <tr>
      <th>Country</th>
      <th>Code</th>
      <th>Visits</th>
      <th>% of Total</th>
    </tr>
    <?php foreach ($countries as $r): 
      // compute percentage (1 decimal place)
      $pct = $total
           ? round($r['cnt'] / $total * 100, 1)
           : 0;
    ?>
      <tr>
        <td><?= htmlspecialchars($r['country_name'] ?: 'Unknown') ?></td>
        <td><?= htmlspecialchars($r['country_code'] ?: '--') ?></td>
        <td><?= htmlspecialchars($r['cnt']) ?></td>
        <td><?= $pct ?>%</td>
      </tr>
    <?php endforeach; ?>
  </table>
</body>
</html>
