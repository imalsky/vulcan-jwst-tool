# Hugging Face Space setup runbook

One-time setup. Cost: $5/mo persistent storage + $0.03/hr CPU Upgrade
hardware while awake (~$22/mo always-on; with a 1 h sleep timer and light
use, typically well under $20/mo total). Prices approximate.

## Fast path (scripted)

After the prerequisite in step 0 below, plus an HF account with billing
configured (hf.co/settings/billing):

    pip install -U huggingface_hub
    hf auth login
    python deploy/hf-space/finish_hf_setup.py

The script replaces manual steps 1-6: it creates both private repos,
pushes the Space files, sets the HF_TOKEN secret, requests storage +
CPU Upgrade + the 1 h sleep timer, and runs the resumable ~7.5 GB data
upload (it stages nothing itself -- run `./upload_data.sh` first if the
staging dir is missing; re-run the script if the upload is interrupted).
Then continue at step 7 (verify). The manual steps below are the fallback
and the reference for what the script does.

## 0. Prerequisite

All three repos pushed to GitHub INCLUDING vulcan-jwst-tool's `deploy/`
directory -- the Space build reads the version pins from the cloned repo's
`deploy/constraints-app.txt` and `deploy/requirements-pandeia.txt`.

## 1. Hugging Face CLI (Mac)

    pip install -U "huggingface_hub[cli]"
    hf auth login

(`huggingface-cli login` on older installs; upload_data.sh accepts either.)

## 2. Create the private dataset repo (browser)

hf.co/new-dataset -> Owner: your account -> Name: `vulcan-jwst-tool-data`
-> Private -> Create. (A different name/owner works too: pass it to the
upload script and set it as a `DATASET_REPO` variable on the Space.)

## 3. Upload the data (Mac; the slow step)

    cd /Users/imalsky/Desktop/Emulators/VULCAN_Project/vulcan-jwst-tool/deploy/hf-space
    ./upload_data.sh

Stages a dereferenced ~7.5 GB copy under ~/Desktop/hf_data_stage, then does
a resumable upload. Re-run the same command if the upload is interrupted.
Delete the staging dir when the Space is confirmed working.

## 4. Create the Space (browser)

hf.co/new-space -> Name: `jwst-tool` -> SDK: Docker (blank template) ->
Private -> CPU basic for now -> Create.

## 5. Push the Space files (Mac)

    cd /Users/imalsky/Desktop/Emulators/VULCAN_Project/vulcan-jwst-tool/deploy/hf-space
    hf upload YOUR-HF-USERNAME/jwst-tool . . --repo-type space

This triggers the first image build (15-30 min; watch the Build logs tab).
The first boot will fail loudly until step 6 is done -- expected.

## 6. Configure the Space (browser, Settings tab)

- Variables and secrets: add secret `HF_TOKEN` = a fine-grained read token
  from hf.co/settings/tokens (read access to the dataset repo). If the
  dataset repo name differs from the default, also add a variable
  `DATASET_REPO` = `owner/name`.
- Storage: add persistent storage, Small (20 GB, $5/mo).
- Hardware: CPU Upgrade (8 vCPU / 32 GB, $0.03/hr). Set sleep time to
  1 h so idle time is not billed.
- Restart the Space.

## 7. Verify

First boot downloads the ~8 GB seed (minutes, HF-internal network), then
the GUI comes up. Check the in-app data status panel (every dataset row
OK), then run the default W39b forward model end-to-end plus one noise
panel (the noise job exercises the pandeia env and the release-match gate).
Wakes from sleep skip the download (markers present).

## 8. Invite users

Space Settings -> add collaborators by HF username (they need free HF
accounts; only the Space needs sharing, not the dataset repo -- data access
uses your HF_TOKEN). Keep the Space private: the entrypoint disables
Streamlit's XSRF/CORS protection to make uploads work behind the Spaces
proxy, which is fine for an invited-user tool but not for a public one.

## Updating code later

Push to GitHub, then Space Settings -> Factory rebuild. Caches on /data
survive rebuilds and self-invalidate by version keys. The exact SHAs baked
into the running image are recorded at /srv/vulcan/BUILD_INFO (shown in the
build logs).

## Known limitations

- Same as the VM deployment: no concurrent-run queue yet; the "legacy"
  Pandeia 3.0 backend is not deployed.
- The image has never been built (no Docker on the Mac) -- the first Space
  build is the real test; read the Build logs on failure.
- Sleep wake-up takes 1-3 min before the GUI responds.
