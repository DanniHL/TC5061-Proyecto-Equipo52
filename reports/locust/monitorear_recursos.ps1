
param(
    [string]$Archivo = "reports/locust/docker_stats.csv"
)

$carpeta = Split-Path $Archivo -Parent
New-Item -ItemType Directory -Force $carpeta | Out-Null

"FechaHora,Contenedor,CPU,MemoriaUsoLimite,MemoriaPorcentaje" |
    Set-Content $Archivo

Write-Host "Registrando consumo de Docker cada 2 segundos."
Write-Host "Archivo: $Archivo"
Write-Host "Presiona Ctrl+C para detener."

while ($true) {
    $fecha = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    docker stats --no-stream --format "{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}" mibici-api mibici-mlflow |
        ForEach-Object {
            "$fecha,$_ " | Add-Content $Archivo
        }

    Start-Sleep -Seconds 2
}