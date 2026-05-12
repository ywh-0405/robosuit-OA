# Baseline Checkpoint

This project was not a Git repository when the checkpoint was created.

Baseline created: 2026-05-11T09:45:30+08:00

## Current Not-Working Version

- Git commit: `c04c106`
- Git tag: `baseline-not-working-20260511`
- Full directory backup:
  `/home/y/Desktop/openarmx_robosuite_example_backups/openarmx_robosuite_example-baseline-20260511-094256`

The backup was verified with:

```bash
diff -qr /home/y/Desktop/openarmx_robosuite_example /home/y/Desktop/openarmx_robosuite_example_backups/openarmx_robosuite_example-baseline-20260511-094256
```

The command exited with status 0 before Git metadata was added.

## Restore Options

Restore tracked files to the checkpoint:

```bash
git -C /home/y/Desktop/openarmx_robosuite_example restore --source baseline-not-working-20260511 -- .
```

Restore the whole directory from the backup:

```bash
cp -a /home/y/Desktop/openarmx_robosuite_example_backups/openarmx_robosuite_example-baseline-20260511-094256/. /home/y/Desktop/openarmx_robosuite_example/
```

Use the full backup if generated files or untracked files must also be restored exactly.
