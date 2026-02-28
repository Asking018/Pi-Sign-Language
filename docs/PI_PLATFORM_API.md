# Pi Platform API Reference (summary)

**Base path:** `https://api.minepi.com/v2`

## Authorization

| Type | Header | Use |
|------|--------|-----|
| **Access token** | `Authorization: Bearer <user access token>` | User-specific resources (e.g. `/me`). Safe from frontend or backend. |
| **Server API Key** | `Authorization: Key <your Server API Key>` | **Server only.** Never expose in client JS. |

---

## Authentication

- **GET /me** — Retrieve user info (scopes, username if consented).  
  Auth: **Access token**.  
  Use: Verify frontend token on your backend (401 if token invalid/tampered).

---

## Payments

### U2A (User-to-App)
- Create payment: **Frontend** `Pi.createPayment()` (JS SDK).
- Then: **onReadyForServerApproval** → your server **POST /payments/{payment_id}/approve** (Server API Key).
- After user signs: **onReadyForServerCompletion** → your server **POST /payments/{payment_id}/complete** with body `{ "txid": "..." }` (Server API Key).

### A2U (App-to-User)
- **POST /payments** — Create A2U payment. Auth: **Server API Key**.  
  Body example:
  ```json
  {
    "payment": {
      "amount": 1,
      "memo": "From app to user",
      "metadata": {},
      "uid": "user's app-specific uid"
    }
  }
  ```

### Common (both types)
- **GET /payments/{payment_id}** — Get payment. Auth: Server API Key.
- **POST /payments/{payment_id}/approve** — Server-side approval. Auth: Server API Key.
- **POST /payments/{payment_id}/complete** — Body `{ "txid": "..." }`. Auth: Server API Key.
- **POST /payments/{payment_id}/cancel** — Cancel payment. Auth: Server API Key.
- **GET /payments/incomplete_server_payments** — List incomplete **A2U** payments. Auth: Server API Key.

---

## Ads

- **GET /ads_network/status/:adId** — Verify rewarded ad status (adId from Pi SDK `displayAd('rewarded')`). Auth: Server API Key.

---

## Resource types (summary)

- **UserDTO:** `uid`, `credentials` (scopes, valid_until), `username` (if scope).
- **PaymentDTO:** `identifier`, `user_uid`, `amount`, `memo`, `metadata`, `from_address`, `to_address`, `direction`, `created_at`, `network`, `status` (developer_approved, transaction_verified, developer_completed, cancelled, user_cancelled), `transaction` (txid, verified, _link).

---

*Full and up-to-date docs: Pi Developer Portal (develop.pi in Pi Browser).*
