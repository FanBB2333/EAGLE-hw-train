#!/bin/bash
# 同时启动所有视频评估脚本
# 每个脚本会在不同的GPU上运行，并在后台执行

echo "开始启动所有视频评估脚本..."

# 获取当前脚本所在的目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 启动各个评估脚本，并在后台运行
echo "启动 eval_videos_0.sh (GPU 0, youcook2)..."
bash "$SCRIPT_DIR/eval_videos_0.sh" > "$SCRIPT_DIR/eval_0.log" 2>&1 &
PID_0=$!

echo "启动 eval_videos_1.sh (GPU 1, activitynet)..."
bash "$SCRIPT_DIR/eval_videos_1.sh" > "$SCRIPT_DIR/eval_1.log" 2>&1 &
PID_1=$!

echo "启动 eval_videos_2.sh (GPU 2, charades)..."
bash "$SCRIPT_DIR/eval_videos_2.sh" > "$SCRIPT_DIR/eval_2.log" 2>&1 &
PID_2=$!

echo "启动 eval_videos_3.sh (GPU 3, qvhighlights)..."
bash "$SCRIPT_DIR/eval_videos_3.sh" > "$SCRIPT_DIR/eval_3.log" 2>&1 &
PID_3=$!

# 记录所有进程ID
echo "所有脚本已启动！"
echo "进程ID:"
echo "  eval_videos_0.sh (youcook2): $PID_0"
echo "  eval_videos_1.sh (activitynet): $PID_1"
echo "  eval_videos_2.sh (charades): $PID_2"
echo "  eval_videos_3.sh (qvhighlights): $PID_3"
echo ""
echo "日志文件:"
echo "  eval_0.log - youcook2 任务日志"
echo "  eval_1.log - activitynet 任务日志"
echo "  eval_2.log - charades 任务日志"
echo "  eval_3.log - qvhighlights 任务日志"
echo ""

# 提供一些有用的命令提示
echo "有用的命令:"
echo "  查看运行状态: ps -f $PID_0 $PID_1 $PID_2 $PID_3"
echo "  实时查看日志: tail -f eval_*.log"
echo "  停止所有进程: kill $PID_0 $PID_1 $PID_2 $PID_3"
echo ""

# 可选：等待所有进程完成
read -p "是否等待所有任务完成？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "等待所有任务完成..."
    wait $PID_0 $PID_1 $PID_2 $PID_3
    echo "所有任务已完成！"
else
    echo "脚本在后台运行中，您可以继续其他工作。"
fi
