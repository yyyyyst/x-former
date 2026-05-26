# Git 常用命令

## 首次推送

```bash
git remote add origin <仓库地址>
git branch -M main
git push -u origin main
```

## 日常提交

```bash
# 查看改了啥
git status
git diff

# 添加 + 提交 + 推送
git add -A
git commit -m "描述改了什么"
git push

# 或者只提交特定文件
git add config/config.yaml config/experiments.md
git commit -m "V3: 解冻layer2, T_0→30, accum→12"
git push
```

## 提交信息格式

```
V1: 基准模型，contrastive=0.1 domain=0.1
V2: contrastive→0.03 domain→0.05，ISLE AUC +15%
V3: freeze_layers→1, T_0→30, accum→12
fix: 修复 joint_collate_fn 混合batch维度不匹配
feat: 新增 experiments.md 调参记录
```

## 查看历史

```bash
git log --oneline            # 简洁版
git log --oneline -10        # 最近10条
git show <commit-hash>       # 看某次提交的详细改动
```

## 撤销

```bash
git checkout -- <file>           # 撤销某个文件的未提交改动
git reset HEAD <file>            # 取消暂存
git reset --soft HEAD~1          # 撤销最近一次 commit，改动保留
```

## 分支

```bash
git branch              # 查看分支
git checkout -b v3      # 新建并切换分支
git checkout main       # 切回主分支
git merge v3            # 合并 v3 到当前分支
```

## 忽略文件

`.gitignore` 中已忽略：
```
logs/
results/
__pycache__/
*.pyc
.pth
.pt
```
