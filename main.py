"""
DIAC∞ Chain API (FastAPI + MongoDB)
-----------------------------------

Features:
- 369 DIAC real coins (fixed supply)
- DVNR (DIAC Virtuales No Reales) as PoW/participation tokens
- DIAC∞ key generation using transcendental windows over π (offset, precision)
- Transaction signing & verification
- DVNR validation (hash DIAC∞ over random number)
- Simple in-memory chain cache + MongoDB persistence (Motor async driver)

Run:
  export MONGO_URI="your mongodb+srv uri"
  uvicorn main:app --reload

Requirements (pip):
  fastapi
  uvicorn[standard]
  motor
  mpmath
  pycryptodome
  python-dotenv  (optional)

Security notes:
- Private keys are encrypted at rest with a per-user symmetric key derived from a passphrase (KDF-quick demo). Replace with stronger KMS/HSM in prod.
- This is demo code; audit before production use.
"""

import os
import json
import secrets
import hashlib
import base64
from typing import List, Optional, Dict, Any

import mpmath
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from Crypto.Util.number import getPrime, inverse
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from dotenv import load_dotenv

# ---------------------------- CONFIG ----------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "diac_chain_db")
SUPPLY_MAX = 369  # real DIAC supply
DVNR_DEFAULT = 10  # free DVNR on signup
PI_MAX_DIGITS = 1_200_000  # safeguard for mpmath precision

# AES for private key encryption (demo only)
KDF_SALT_SIZE = 16

# ---------------------------- UTILS -----------------------------

def kdf_from_passphrase(passphrase: str, salt: bytes) -> bytes:
    """Derive 32-byte key from passphrase using simple repeated SHA256 (demo only)."""
    key = passphrase.encode() + salt
    for _ in range(100000):
        key = hashlib.sha256(key).digest()
    return key  # 32 bytes

def aes_encrypt(data: bytes, key: bytes) -> bytes:
    iv = secrets.token_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ct = cipher.encrypt(pad(data, 16))
    return iv + ct

def aes_decrypt(blob: bytes, key: bytes) -> bytes:
    iv, ct = blob[:16], blob[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), 16)


def get_N_from_pi(offset: int, precision: int) -> int:
    if offset < 0 or precision <= 0:
        raise ValueError("offset/precision invalid")
    need = offset + precision + 10
    if need > PI_MAX_DIGITS:
        raise ValueError("Requested too many digits of pi")
    mpmath.mp.dps = need
    pi_str = str(mpmath.mp.pi)[2:]  # remove "3."
    slice_str = pi_str[offset:offset + precision]
    if len(slice_str) < precision:
        # fallback to pad with random? safer to error
        raise ValueError("Not enough digits computed")
    return int(slice_str)


def generate_diac_keypair():
    x = secrets.randbelow(2**256 - 2) + 2
    p = getPrime(512)
    offset = secrets.randbelow(10**6)
    precision = secrets.randbelow(181) + 20  # 20..200
    N = get_N_from_pi(offset, precision)
    xN = (x * N) % p
    xN_inv = inverse(xN, p)
    pubkey = (xN - xN_inv) % p
    hashN = hashlib.sha256(str(N).encode()).hexdigest()
    private = {"x": x, "offset": offset, "precision": precision, "p": p}
    public = {"pubkey": str(pubkey), "p": str(p), "hashN": hashN}
    return private, public


def derive_symm_key(pubkey: str, p: str, hashN: str, nonce: bytes) -> bytes:
    base = pubkey + p + hashN + nonce.hex()
    return hashlib.sha256(base.encode()).digest()


def sign_message(message: str, private: Dict[str, Any]) -> Dict[str, Any]:
    N = get_N_from_pi(private['offset'], private['precision'])
    xN = (private['x'] * N) % private['p']
    xN_inv = inverse(xN, private['p'])
    pubkey = (xN - xN_inv) % private['p']
    hashN = hashlib.sha256(str(N).encode()).hexdigest()
    to_sign = f"{message}|{pubkey}|{private['p']}|{hashN}"
    sig = hashlib.sha256(to_sign.encode()).hexdigest()
    return {
        "message": message,
        "pubkey": str(pubkey),
        "p": str(private['p']),
        "hashN": hashN,
        "sig": sig
    }


def verify_signature(signed_msg: Dict[str, Any]) -> bool:
    to_check = f"{signed_msg['message']}|{signed_msg['pubkey']}|{signed_msg['p']}|{signed_msg['hashN']}"
    return hashlib.sha256(to_check.encode()).hexdigest() == signed_msg['sig']


def diac_infinite_hash(number: int, priv: Dict[str, Any]) -> str:
    N = get_N_from_pi(priv['offset'], priv['precision'])
    xN = (priv['x'] * N) % priv['p']
    xN_inv = inverse(xN, priv['p'])
    pubkey = (xN - xN_inv) % priv['p']
    base = f"{number}|{pubkey}|{priv['p']}"
    return hashlib.sha256(base.encode()).hexdigest()

# ---------------------------- MODELS ----------------------------

class CreateUserReq(BaseModel):
    passphrase: str = Field(..., description="Used to encrypt your private key")

class CreateUserResp(BaseModel):
    pubkey: Dict[str, str]
    dvnr_list: List[int]

class BalanceResp(BaseModel):
    diac: int
    dvnr: int

class SendDiacReq(BaseModel):
    from_pubkey: str
    to_pubkey: str
    amount: int
    passphrase: str

class UseDVNRReq(BaseModel):
    pubkey: str
    dvnr_number: int

class ValidateDVNRReq(BaseModel):
    pubkey: str
    dvnr_number: int
    passphrase: str

class TxRecord(BaseModel):
    type: str
    tx: Dict[str, Any]

# ------------------------ FASTAPI SETUP -------------------------

app = FastAPI(title="DIAC∞ Chain API", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# Collections
col_users = db.users
col_balances = db.balances
col_chain = db.chain
col_dvnr = db.dvnr
col_meta = db.meta  # stores supply, genesis flag

# ------------------------- STARTUP TASK -------------------------

@app.on_event("startup")
async def init_db():
    # Ensure indexes
    await col_users.create_index("pubkey.pubkey", unique=True)
    await col_balances.create_index("pubkey", unique=True)
    await col_dvnr.create_index([("pubkey", 1), ("dvnr", 1)], unique=True)

    meta = await col_meta.find_one({"_id": "meta"})
    if not meta:
        # First run: create empty meta doc
        await col_meta.insert_one({"_id": "meta", "genesis": False, "supply": SUPPLY_MAX})

    meta = await col_meta.find_one({"_id": "meta"})
    if not meta.get("genesis"):
        # Wait until a user is created to assign genesis
        print("Genesis not created yet. Will assign to first user created.")

# --------------------------- HELPERS ----------------------------

async def encrypt_private_key(private: Dict[str, Any], passphrase: str) -> Dict[str, str]:
    salt = secrets.token_bytes(KDF_SALT_SIZE)
    key = kdf_from_passphrase(passphrase, salt)
    blob = aes_encrypt(json.dumps(private).encode(), key)
    return {
        "salt": base64.b64encode(salt).decode(),
        "blob": base64.b64encode(blob).decode()
    }

async def decrypt_private_key(enc: Dict[str, str], passphrase: str) -> Dict[str, Any]:
    salt = base64.b64decode(enc['salt'])
    blob = base64.b64decode(enc['blob'])
    key = kdf_from_passphrase(passphrase, salt)
    data = aes_decrypt(blob, key)
    return json.loads(data.decode())

async def get_balance(pubkey: str) -> int:
    bal_doc = await col_balances.find_one({"pubkey": pubkey})
    return bal_doc.get("diac", 0) if bal_doc else 0

async def set_balance(pubkey: str, amount: int):
    await col_balances.update_one({"pubkey": pubkey}, {"$set": {"diac": amount}}, upsert=True)

async def add_balance(pubkey: str, delta: int):
    await col_balances.update_one({"pubkey": pubkey}, {"$inc": {"diac": delta}}, upsert=True)

async def count_dvnr(pubkey: str) -> int:
    return await col_dvnr.count_documents({"pubkey": pubkey, "used": False})

async def insert_tx(tx: Dict[str, Any]):
    await col_chain.insert_one({"type": "tx", "tx": tx})

async def create_genesis_for(pubkey_doc: Dict[str, Any]):
    # Give all supply to first user
    await set_balance(pubkey_doc['pubkey']['pubkey'], SUPPLY_MAX)
    await col_chain.insert_one({
        "type": "genesis",
        "to": pubkey_doc['pubkey']['pubkey'],
        "amount": SUPPLY_MAX
    })
    await col_meta.update_one({"_id": "meta"}, {"$set": {"genesis": True}})

# --------------------------- ENDPOINTS --------------------------

@app.post("/create_user", response_model=CreateUserResp)
async def create_user(req: CreateUserReq):
    private, public = generate_diac_keypair()
    dvnr_list = [secrets.randbelow(10**12) for _ in range(DVNR_DEFAULT)]

    enc_priv = await encrypt_private_key(private, req.passphrase)

    user_doc = {
        "pubkey": public,
        "priv_enc": enc_priv,
        "dvnr": [{"dvnr": d, "used": False} for d in dvnr_list]
    }
    try:
        await col_users.insert_one(user_doc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"User insert error: {e}")

    # if no genesis, assign to this user
    meta = await col_meta.find_one({"_id": "meta"})
    if not meta.get("genesis"):
        await create_genesis_for(user_doc)

    # Persist dvnr separately for quick queries
    if dvnr_list:
        await col_dvnr.insert_many([{ "pubkey": public['pubkey'], "dvnr": d, "used": False } for d in dvnr_list])

    return CreateUserResp(pubkey=public, dvnr_list=dvnr_list)


@app.get("/get_balance/{pubkey}", response_model=BalanceResp)
async def get_balance_endpoint(pubkey: str):
    diac = await get_balance(pubkey)
    dvnr = await count_dvnr(pubkey)
    return BalanceResp(diac=diac, dvnr=dvnr)


@app.post("/send_diac")
async def send_diac(req: SendDiacReq):
    # load sender priv
    user = await col_users.find_one({"pubkey.pubkey": req.from_pubkey})
    if not user:
        raise HTTPException(404, "Sender not found")
    try:
        priv = await decrypt_private_key(user['priv_enc'], req.passphrase)
    except Exception:
        raise HTTPException(403, "Bad passphrase")

    sender_balance = await get_balance(req.from_pubkey)
    if sender_balance < req.amount:
        raise HTTPException(400, "Insufficient DIAC")

    # make tx
    message = f"{req.from_pubkey}->{req.to_pubkey}:{req.amount}"
    tx = sign_message(message, priv)
    if not verify_signature(tx):
        raise HTTPException(400, "Signature failed")

    # update balances atomically-ish (2 ops; Mongo doesn't need transaction here for demo)
    await add_balance(req.from_pubkey, -req.amount)
    await add_balance(req.to_pubkey, req.amount)
    await insert_tx(tx)

    return {"status": "ok", "tx": tx}


@app.post("/use_dvnr")
async def use_dvnr(req: UseDVNRReq):
    dv = await col_dvnr.find_one({"pubkey": req.pubkey, "dvnr": req.dvnr_number, "used": False})
    if not dv:
        raise HTTPException(404, "DVNR not available or already used")
    await col_dvnr.update_one({"_id": dv["_id"]}, {"$set": {"used": True}})
    return {"status": "used", "dvnr": req.dvnr_number}


@app.post("/validate_dvnr")
async def validate_dvnr(req: ValidateDVNRReq):
    # check DVNR exists and not used
    dv = await col_dvnr.find_one({"pubkey": req.pubkey, "dvnr": req.dvnr_number, "used": False})
    if not dv:
        raise HTTPException(404, "DVNR not available or already used")

    user = await col_users.find_one({"pubkey.pubkey": req.pubkey})
    if not user:
        raise HTTPException(404, "User not found")

    try:
        priv = await decrypt_private_key(user['priv_enc'], req.passphrase)
    except Exception:
        raise HTTPException(403, "Bad passphrase")

    # hash with DIAC∞
    h = diac_infinite_hash(req.dvnr_number, priv)

    # simple winning rule: first 4 hex chars == '0000'
    if h[:4] == '0000':
        # award 1 DIAC from user's own balance? or from a prize pool? we'll use global reserve if available
        # for demo: from nowhere if supply remains; but supply is fixed, so we need pool. Let's skip supply minting: just give prestige.
        # Alternative: move 1 DIAC from a "treasury" pubkey (the genesis holder) if exists
        meta = await col_meta.find_one({"_id": "meta"})
        treasury_pub = await col_chain.find_one({"type": "genesis"})
        if treasury_pub:
            t_pub = treasury_pub['to']
            if await get_balance(t_pub) > 0:
                await add_balance(t_pub, -1)
                await add_balance(req.pubkey, 1)
                prize = 1
            else:
                prize = 0
        else:
            prize = 0

        await col_dvnr.update_one({"_id": dv["_id"]}, {"$set": {"used": True, "win": True}})
        return {"result": "win", "hash": h, "prize_diac": prize}
    else:
        await col_dvnr.update_one({"_id": dv["_id"]}, {"$set": {"used": True, "win": False}})
        return {"result": "lose", "hash": h}


@app.get("/chain", response_model=List[Dict[str, Any]])
async def get_chain(limit: int = 50):
    cursor = col_chain.find({}).sort("_id", -1).limit(limit)
    return [doc async for doc in cursor]


@app.get("/user/{pubkey}")
async def get_user(pubkey: str):
    user = await col_users.find_one({"pubkey.pubkey": pubkey}, {"priv_enc": 0})
    if not user:
        raise HTTPException(404, "Not found")
    return user

# --------------------------- END FILE ---------------------------
