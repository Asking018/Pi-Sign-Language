# Pi Client SDK Reference (summary)

Load on every page that uses the SDK:

```html
<script src="https://sdk.minepi.com/pi-sdk.js"></script>
<script>Pi.init({ version: "2.0" });</script>
```

## Init config

| Key | Type | Description |
|-----|------|-------------|
| `version` | string, **required** | e.g. `"2.0"` — compatibility with future SDK versions. |
| `sandbox` | boolean, optional | `true` for sandbox (use with https://sandbox.minepi.com and dev URL in Developer Portal). |

Sandbox example: `Pi.init({ version: "2.0", sandbox: true });`

---

## Authentication

**Warning:** Use SDK user info only for UI (e.g. display username). On the backend, use the Platform API as source of truth (e.g. **GET /me** with the user’s access token).

```ts
Pi.authenticate(scopes: Scope[], onIncompletePaymentFound?: (payment: PaymentDTO) => void): Promise<AuthResult>
```

**AuthResult:**
- `accessToken: string`
- `user: { uid: string; username?: string }`

**Scopes:** `"username"` | `"payments"` | `"wallet_address"`

| Field | Scope |
|-------|--------|
| uid | (none) |
| username | username |
| payments | payments |
| wallet_address | wallet_address |

**onIncompletePaymentFound(payment):** Called when the user has an incomplete payment (tx submitted but `status.developer_completed === false`). You must complete that payment (e.g. send to your server and call `/complete`) before starting a new payment.

---

## Payments (U2A)

```ts
Pi.createPayment(paymentData: PaymentData, callbacks: PaymentCallbacks): void
```

**PaymentData:** `{ amount: number, memo: string, metadata: object }`

**PaymentCallbacks:**

| Callback | Signature | Use |
|----------|------------|-----|
| onReadyForServerApproval | `(paymentId: string) => void` | Send `paymentId` to your backend → **POST /payments/{id}/approve**. May be retried ~every 10s on failure. |
| onReadyForServerCompletion | `(paymentId: string, txid: string) => void` | Send `paymentId` + `txid` to your backend → **POST /payments/{id}/complete**. May be retried ~every 10s on failure. |
| onCancel | `(paymentId: string) => void` | User cancelled or blocking situation (e.g. insufficient funds, concurrent payment). |
| onError | `(error: Error, payment?: PaymentDTO) => void` | Error; second arg present if payment was created. |

**Concurrent payments:** If a new payment is created while one is open: (1) if user hasn’t submitted tx yet, open payment is cancelled; (2) if user already submitted tx, new payment is rejected (onError) and `onIncompletePaymentFound` is called with the existing payment — complete it first.

---

## Types (summary)

- **PaymentDTO:** identifier, user_uid, amount, memo, metadata, from_address, to_address, direction, created_at, network, status (developer_approved, transaction_verified, developer_completed, cancelled, user_cancelled), transaction (txid, verified, _link).
- **Direction:** `"user_to_app"` | `"app_to_user"`.
- **AppNetwork:** `"Pi Network"` | `"Pi Testnet"`.
- **Scope:** `"username"` | `"payments"` | `"wallet_address"`.

---

## Native features

```ts
Pi.nativeFeaturesList(): Promise<NativeFeature[]>
```
**NativeFeature:** `"inline_media"` | `"request_permission"` | `"ad_network"`.

---

## Share

```ts
Pi.openShareDialog(title: string, message: string): void
```

---

## Ads

**Show ad:** `Pi.Ads.showAd(adType: "interstitial" | "rewarded"): Promise<ShowAdResponse>`

- **rewarded** response may include `adId` — verify rewarded status via Platform API **GET /ads_network/status/:adId** before granting rewards (required if approved for Pi Developer Ad Network).
- Results: AD_REWARDED, AD_CLOSED, AD_DISPLAY_ERROR, AD_NETWORK_ERROR, AD_NOT_AVAILABLE, ADS_NOT_SUPPORTED, USER_UNAUTHENTICATED (rewarded only).

**Check ready:** `Pi.Ads.isAdReady(adType): Promise<{ type, ready }>`  
**Request ad:** `Pi.Ads.requestAd(adType): Promise<{ type, result: "AD_LOADED" | "AD_FAILED_TO_LOAD" | "AD_NOT_AVAILABLE" }>`

---

## Open URL in system browser

```ts
Pi.openUrlInSystemBrowser(url: string): Promise<void>
```
Rejects with: "Failed to open URL" | "No minimal requirements" | "Unexpected error".

---

*Full docs: Pi Developer Portal (develop.pi in Pi Browser).*
