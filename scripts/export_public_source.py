"""Export the current source tree into a new repository without inherited history."""
from pathlib import Path
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / 'public-release' / 'ophthalmic-imaging-platform'
    if destination.exists():
        raise SystemExit('Export already exists; choose a new destination before exporting again.')
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root
    ).decode('utf-8').split('\0')
    allowed = {'.py', '.toml', '.md', '.txt', '.json', '.jsx', '.js', '.css',
               '.html', '.ini', '.mako', '.cmd', '.ps1', '.svg', '.example'}
    sources = []
    for name in sorted(set(filter(None, names))):
        source = root / name
        if not source.exists():
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise SystemExit('Refusing an external source path.')
        if name.startswith('public-release/'):
            continue
        if source.suffix not in allowed and source.name not in {'.gitignore', 'LICENSE'}:
            raise SystemExit(f'Unexpected file type: {name}')
        source.read_text(encoding='utf-8')  # Reject binary artifacts.
        if source.name.startswith('.env') and source.name != '.env.example':
            raise SystemExit('Refusing a real environment file.')
        sources.append((source, destination / name))
    destination.mkdir(parents=True)
    for source, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    subprocess.run(['git', 'init', '--initial-branch=main', str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'add', '.'], check=True)
    print(f'Exported {len(sources)} source files. No original Git history or remote was copied.')


if __name__ == '__main__':
    main()
