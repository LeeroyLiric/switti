# sh_scripts/run_gen_seeds.ps1

$CheckpointChoice = "2"   # 1 = iter 100, 2 = iter 200
$Prompt = ""              # пусто = default prompt

$Seeds = @(1, 42, 123, 697, 850)

foreach ($Seed in $Seeds) {
    Write-Host ""
    Write-Host "=== Generating with checkpoint choice $CheckpointChoice, seed $Seed ==="

    if ([string]::IsNullOrWhiteSpace($Prompt)) {
        python .\checkpoint_gen.py --checkpoint $CheckpointChoice --seed $Seed
    }
    else {
        python .\checkpoint_gen.py --checkpoint $CheckpointChoice --prompt $Prompt --seed $Seed
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Error on seed $Seed. Stopping."
        break
    }
}

Write-Host ""
Write-Host "Done."
