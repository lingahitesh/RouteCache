# Check if the FastAPI server is running

Write-Host "Checking if server is running on http://localhost:8000..." -ForegroundColor Yellow

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction Stop
    Write-Host "SUCCESS: Server is running!" -ForegroundColor Green
    Write-Host "  Status: $($response.StatusCode)" -ForegroundColor Green
    exit 0
} catch {
    Write-Host "ERROR: Server is NOT running!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please start the server first:" -ForegroundColor Yellow
    Write-Host "  1. Open a terminal" -ForegroundColor White
    Write-Host "  2. cd C:\Users\KIIT\Desktop\Projects\RouteCache" -ForegroundColor White
    Write-Host "  3. .\.venv\Scripts\Activate.ps1" -ForegroundColor White  
    Write-Host "  4. uvicorn app.main:app --reload" -ForegroundColor White
    Write-Host ""
    exit 1
}
