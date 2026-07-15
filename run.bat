@echo off
chcp 65001 >nul
cd /d D:\Xunlong

echo ========================================
echo  寻龙点穴引擎 - Xunlong Engine
echo ========================================

echo [1/4] 清理旧进程 ...
powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force"
timeout /t 2 >nul

echo [2/4] 启动后端服务 ...
powershell -Command "Start-Process -FilePath 'D:\Xunlong\engine\.venv\Scripts\python.exe' -ArgumentList 'D:\Xunlong\engine\run_server.py' -WorkingDirectory 'D:\Xunlong' -RedirectStandardOutput 'D:\Xunlong\engine\server.log' -RedirectStandardError 'D:\Xunlong\engine\server.err.log' -NoNewWindow -PassThru | Select-Object Id"
timeout /t 5 >nul

echo [3/4] 验证服务 ...
echo --- server.log ---
type "D:\Xunlong\engine\server.log"
echo --- server.err.log ---
type "D:\Xunlong\engine\server.err.log"

echo [4/4] 打开前端 ...
start http://127.0.0.1:8765/

echo ========================================
echo  后端: http://127.0.0.1:8765
echo  健康: GET /api/health
echo ========================================
echo  按任意键停止服务 ... ^
powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force"
pause >nul
