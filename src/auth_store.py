"""Local credentials with PBKDF2 hashes; no default/public passwords."""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import re
from pathlib import Path

ITERATIONS = 350_000

class UserStore:
    def __init__(self, app_root: Path):
        root = Path(app_root) / 'config'
        root.mkdir(parents=True, exist_ok=True)
        self.file = root / 'users.json'

    def _load(self) -> dict:
        if not self.file.exists():
            return {}
        obj = json.loads(self.file.read_text(encoding='utf-8'))
        return obj if isinstance(obj, dict) else {}

    def _save(self, obj: dict) -> None:
        tmp = self.file.with_suffix('.tmp')
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(self.file)
        try:
            os.chmod(self.file, 0o600)
        except OSError:
            pass

    @property
    def has_users(self) -> bool:
        return bool(self._load())

    def add_user(self, username: str, password: str, role: str) -> None:
        name = username.strip().lower()
        if not re.fullmatch(r'[a-z0-9_.-]{3,32}', name):
            raise ValueError('Usuario: 3-32 caracteres (letras, números, _, . o -).')
        if len(password) < 8:
            raise ValueError('La contraseña necesita al menos 8 caracteres.')
        if role not in ('engineer', 'operator'):
            raise ValueError('Rol inválido.')
        users = self._load()
        if not users and role != 'engineer':
            raise ValueError('La primera cuenta debe ser de ingeniero.')
        if name in users:
            raise ValueError('El nombre de usuario ya existe.')
        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, ITERATIONS)
        users[name] = {'role': role, 'salt': salt.hex(), 'hash': digest.hex(), 'iterations': ITERATIONS}
        self._save(users)

    def login(self, username: str, password: str) -> str | None:
        record = self._load().get(username.strip().lower())
        if not record:
            return None
        try:
            digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(record['salt']), int(record['iterations']))
            return record['role'] if hmac.compare_digest(digest, bytes.fromhex(record['hash'])) else None
        except (ValueError, TypeError, KeyError):
            return None

    def list_users(self) -> list[tuple[str, str]]:
        return [(n, data['role']) for n, data in sorted(self._load().items())]
