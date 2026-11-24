# MCP 与桌面引擎接入指南（PhotoCraft + LightCraft）

`retouch-workbench` 通过 **ArtCraft PhotoCraft**（分层修图、蒙版、液化、画笔、PSD/PCRAFT 无损导出）和 **ArtCraft LightCraft**（RAW 批量显影、选片、32-bit 色彩分级）的原生 `stdio` MCP Server 与 Agent 协作。

## 客户端 MCP 配置示例

将 `%LOCALAPPDATA%` 替换为你的本地 AppData 路径，在支持 MCP 的客户端（Codex / Antigravity / Claude Code 等）中添加如下条目：

```json
{
  "mcpServers": {
    "photocraft": {
      "command": "%LOCALAPPDATA%/Programs/ArtCraft/PhotoCraft/photocraft-cli.exe",
      "args": ["mcp", "--compact", "--read-root", "C:/", "--write-root", "D:/ps"]
    },
    "lightcraft": {
      "command": "%LOCALAPPDATA%/Programs/ArtCraft/LightCraft/lightcraft-cli.exe",
      "args": ["mcp", "--compact", "--library", "D:/ps/.lightcraft-library"]
    }
  }
}
```

- `--read-root`：允许读取原始拍摄素材的路径范围（只读不改）。
- `--write-root`：限制工程保存与成片导出写入的工作目录，避免误写系统区。

## 实时桌面协作模式（可选）

如果希望在 Agent 跑命令的同时在桌面端实时看到图层画面与交互检查，可以先带控制端口启动 PhotoCraft 桌面版，再用 `--bridge` 接入：

```powershell
& "$env:LOCALAPPDATA\Programs\ArtCraft\PhotoCraft\photocraft.exe" --control 19234
& "$env:LOCALAPPDATA\Programs\ArtCraft\PhotoCraft\photocraft-cli.exe" mcp --bridge 127.0.0.1:19234
```

## 常用 CLI 核验命令

```powershell
# 查看图片真实编码、尺寸与位深（防 .png 假壳）
node scripts\inspect-image.mjs inspect path\to\photo.png

# 查看 .pcraft / .psd 分层树与警告
& "$env:LOCALAPPDATA\Programs\ArtCraft\PhotoCraft\photocraft-cli.exe" info path\to\project.pcraft

# 检查本地开源 AI 侧车、ONNX CUDA 与模型目录状态
& "$env:LOCALAPPDATA\RetouchTools\venv\Scripts\python.exe" scripts\retouch-ai.py doctor
```
