#!/bin/zsh
# Sync src/ to the box. No -a (fuse mount rejects chown), no --delete (box has run scripts we want).
cd /Users/jonas/Development/secret_loyalities/breadth-vs-detectability
rsync -rltz --no-owner --no-group --no-perms \
  -e "ssh -p 22091 -i $HOME/.ssh/jmhgen_vast -o StrictHostKeyChecking=no" \
  --exclude '__pycache__' src/ root@69.30.85.132:/workspace/slbd/
