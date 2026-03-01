#!/usr/bin/env bash
# 将仓库中的 skill 软链接到 ~/.claude/skills/
# 用法: bash setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILLS_DIR="$HOME/.claude/skills"

mkdir -p "$SKILLS_DIR"

for skill in "$SCRIPT_DIR"/skill/*/; do
    name=$(basename "$skill")
    target="$SKILLS_DIR/$name"
    if [ -L "$target" ]; then
        echo "更新: $target -> $skill"
        rm "$target"
    elif [ -e "$target" ]; then
        echo "跳过: $target 已存在且不是软链接，请手动处理"
        continue
    else
        echo "创建: $target -> $skill"
    fi
    ln -s "$skill" "$target"
done

echo "done"
