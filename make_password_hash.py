"""Generate a password hash for the auth config:  python make_password_hash.py"""
import getpass

from pwhash import hash_password

pw = getpass.getpass("Password: ")
if len(pw) < 12:
    raise SystemExit("Use at least 12 characters.")
if pw != getpass.getpass("Repeat: "):
    raise SystemExit("Passwords do not match.")
print(hash_password(pw))
