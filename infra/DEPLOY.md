# flagOR on Azure — `ssh johnypuzzles.<region>.cloudapp.azure.com` from any computer

Layout on the VM:
- **port 22** → the puzzle (asyncssh, no password, like `ssh augustalabs.ai`)
- **port 2222** → your admin ssh (key-based, user `azureuser`)
- app in `/opt/flagor`, systemd service `flagor`, model cache `/var/lib/flagor/hf` (runs offline)

## One-time (Mac)
    brew install azure-cli
    az login                                  # opens the browser; pick the Azure for Students subscription

## Create + deploy (Mac, from backup/)
    bash infra/azure_create.sh                # prints the FQDN, e.g. johnypuzzles.westeurope.cloudapp.azure.com
    bash infra/bundle.sh
    bash infra/push.sh <FQDN>                 # ~10 min: packages, CPU torch, model download, service

If creation fails with *RequestDisallowedByPolicy* (student subscriptions allow only some regions)
or a size is unavailable, retry with e.g. `LOC=northeurope`, `LOC=swedencentral`, `LOC=francecentral`
and/or `SIZE=Standard_B2s`. If the DNS label is taken, use `DNS=johnypuzzles-pt`.

## Use
    ssh <FQDN>                                # the puzzle (anyone)
    ssh -p 2222 azureuser@<FQDN>              # you (admin)
    ssh -p 2222 azureuser@<FQDN> 'sudo journalctl -u flagor -f'    # live logs

## Update code / adapter
    bash infra/bundle.sh && bash infra/redeploy.sh <FQDN>

## Save credit
    az vm deallocate -g johnypuzzles-rg -n johnypuzzles-vm   # stops compute billing (IP/DNS kept)
    az vm start      -g johnypuzzles-rg -n johnypuzzles-vm
    az group delete  -n johnypuzzles-rg                # delete everything

## Custom domain (optional, e.g. flagor.pt)
Buy the domain, add an **A record** → the VM's public IP (static). Then set the brief's host:
`sudo sed -i 's/FLAGOR_PUBLIC_HOST=.*/FLAGOR_PUBLIC_HOST=flagor.pt/' /etc/systemd/system/flagor.service && sudo systemctl daemon-reload && sudo systemctl restart flagor`

## On-demand (cheapest way to run the full box)
    bash infra/on.sh          # start; puzzle ready ~2 min later
    bash infra/off.sh         # deallocate; VM billing stops (disk + IP ≈ €6/month remain)
    bash infra/nightly_off.sh # once: auto-deallocate every night, in case you forget
Never "turn off" with `sudo shutdown` inside the VM — that keeps billing. Portal **Stop** = deallocate (fine).

## Own domain (e.g. johnyriddles.ai)
1. Buy it at a registrar (Porkbun / Namecheap / Cloudflare). Azure student credit can't pay for domains.
2. DNS: add an **A record**, host `@` → the VM IP (`az vm show -d -g johnypuzzles-rg -n johnypuzzles-vm --query publicIps -o tsv`).
3. `dig +short johnyriddles.ai` shows the IP → `bash infra/set_domain.sh <FQDN> johnyriddles.ai`
   (or on a fresh deploy: `bash infra/push.sh <FQDN> johnyriddles.ai`).
