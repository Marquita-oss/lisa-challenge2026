# setup_env.ps1
# Configura las variables de entorno de nnU-Net v2 de forma permanente
# para el usuario actual (persisten entre sesiones).
#
# Uso:
#   .\setup_env.ps1                     # configura en el directorio del proyecto
#   .\setup_env.ps1 -Scope Machine      # para todos los usuarios (requiere admin)
#
# Verificar después de ejecutar:
#   [System.Environment]::GetEnvironmentVariable("nnUNet_raw", "User")

param(
    [ValidateSet("User", "Machine")]
    [string]$Scope = "User"
)

# ---------------------------------------------------------------------------
# Rutas base del proyecto
# ---------------------------------------------------------------------------
$ProjectRoot = "C:\Users\rmarcar\Desktop\lisa-challenge2026"
$Workspace   = "$ProjectRoot\nnunet_workspace"

$Vars = @{
    "nnUNet_raw"          = "$Workspace\nnUNet_raw"
    "nnUNet_preprocessed" = "$Workspace\nnUNet_preprocessed"
    "nnUNet_results"      = "$Workspace\nnUNet_results"
}

# ---------------------------------------------------------------------------
# Crear directorios si no existen
# ---------------------------------------------------------------------------
foreach ($path in $Vars.Values) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Write-Host "Creado: $path"
    }
}

# ---------------------------------------------------------------------------
# Registrar variables de entorno de forma permanente
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Configurando variables de entorno (Scope: $Scope)..."
Write-Host ""

foreach ($entry in $Vars.GetEnumerator()) {
    [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, $Scope)
    # También disponible en la sesión actual
    [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    Write-Host "  SET $($entry.Key)"
    Write-Host "      = $($entry.Value)"
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Verificación inmediata
# ---------------------------------------------------------------------------
Write-Host "--- Verificación ---"
$allOk = $true
foreach ($entry in $Vars.GetEnumerator()) {
    $stored = [System.Environment]::GetEnvironmentVariable($entry.Key, $Scope)
    if ($stored -eq $entry.Value) {
        Write-Host "  [OK]  $($entry.Key)"
    } else {
        Write-Host "  [FAIL] $($entry.Key) — valor almacenado: '$stored'"
        $allOk = $false
    }
}

Write-Host ""
if ($allOk) {
    Write-Host "Variables configuradas correctamente."
    Write-Host ""
    Write-Host "IMPORTANTE: abrir una nueva terminal para que tomen efecto en futuras sesiones."
    Write-Host "En la sesión actual ya están disponibles."
    Write-Host ""
    Write-Host "Próximo paso:"
    Write-Host "  pip install nnunetv2"
    Write-Host "  nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity"
} else {
    Write-Host "Hubo errores. Si usaste Scope=Machine, ejecutar PowerShell como Administrador."
}
