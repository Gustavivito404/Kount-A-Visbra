"""Local Kount-A accounts; PBKDF2 hashes and owner-controlled user management."""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import re
from pathlib import Path

ITERATIONS = 350_000
ROLES = ('owner', 'engineer', 'operator')

class UserStore:
    def __init__(self, app_root: Path):
        folder = Path(app_root) / 'config'
        folder.mkdir(parents=True, exist_ok=True)
        self.file = folder / 'users.json'

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

    @property
    def has_owner(self) -> bool:
        return any(u.get('role') == 'owner' and u.get('enabled', True)
                   for u in self._load().values())

    @staticmethod
    def _name(username: str) -> str:
        name = username.strip().lower()
        if not re.fullmatch(r'[a-z0-9_.-]{3,32}', name):
            raise ValueError('Usuario: 3–32 caracteres (letras, números, _, . o -).')
        return name

    @staticmethod
    def _hash_password(password: str) -> dict:
        if len(password) < 8:
            raise ValueError('La contraseña necesita al menos 8 caracteres.')
        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, ITERATIONS)
        return {'salt': salt.hex(), 'hash': digest.hex(), 'iterations': ITERATIONS}

    def add_user(self, username: str, password: str, role: str) -> None:
        name = self._name(username)
        if role not in ROLES:
            raise ValueError('Rol inválido.')
        users = self._load()
        if not users and role != 'owner':
            raise ValueError('La primera cuenta debe ser del dueño.')
        if users and not self.has_owner:
            raise ValueError('Primero debe habilitarse la cuenta del dueño.')
        if name in users:
            raise ValueError('El nombre de usuario ya existe.')
        users[name] = {'role': role, 'enabled': True, **self._hash_password(password)}
        self._save(users)

    def login(self, username: str, password: str) -> str | None:
        record = self._load().get(username.strip().lower())
        if not record or not record.get('enabled', True):
            return None
        try:
            digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'),
                                         bytes.fromhex(record['salt']), int(record['iterations']))
            return record['role'] if hmac.compare_digest(digest, bytes.fromhex(record['hash'])) else None
        except (ValueError, TypeError, KeyError):
            return None

    def promote_legacy_engineer(self, username: str, password: str) -> bool:
        """Upgrade an existing v15 installation ONLY after verifying that engineer's password."""
        name = username.strip().lower()
        if self.has_owner or self.login(name, password) != 'engineer':
            return False
        users = self._load()
        if users[name]['role'] != 'engineer':
            return False
        users[name]['role'] = 'owner'
        self._save(users)
        return True

    def list_users(self) -> list[tuple[str, str]]:
        return [(name, data['role']) for name, data in sorted(self._load().items())]

    def user_details(self) -> list[dict]:
        return [{'username': name, 'role': data['role'], 'enabled': data.get('enabled', True)}
                for name, data in sorted(self._load().items())]

    def update_user(self, username: str, *, new_username: str | None = None,
                    role: str | None = None, enabled: bool | None = None,
                    password: str | None = None) -> str:
        """Update a user; never delete or demote/disable the last active owner."""
        users = self._load()
        old = username.strip().lower()
        if old not in users:
            raise ValueError('No existe el usuario.')
        new = self._name(new_username) if new_username is not None else old
        if new != old and new in users:
            raise ValueError('El nuevo nombre de usuario ya existe.')
        record = dict(users[old])
        if role is not None:
            if role not in ROLES:
                raise ValueError('Rol inválido.')
            record['role'] = role
        if enabled is not None:
            record['enabled'] = bool(enabled)
        if password is not None:
            record.update(self._hash_password(password))
        was_last_owner = (users[old].get('role') == 'owner' and users[old].get('enabled', True)
                          and sum(1 for u in users.values() if u.get('role') == 'owner'
                                  and u.get('enabled', True)) == 1)
        if was_last_owner and (record['role'] != 'owner' or not record.get('enabled', True)):
            raise ValueError('No se puede desactivar ni cambiar el rol del último dueño activo.')
        del users[old]
        users[new] = record
        self._save(users)
        return new
