from typing import List, Dict, Optional
import requests

class CVETool:
    """
    CVE Tool 🛠️
    - Search CVEs
    - Get CVE details
    - Get related exploits
    - Get patch info (safe, skips broken links)
    """

    BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    def search_cves(self, keyword: str, limit: int = 10) -> List[Dict]:
        """
        Search CVEs by keyword (e.g., software name, vulnerability type)
        """
        print("search_cves running from CVETool")
        params = {"keywordSearch": keyword, "resultsPerPage": limit}
        response = requests.get(self.BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get("vulnerabilities", [])
        return []

    def get_cve_details(self, cve_id: str) -> Optional[Dict]:
        """
        Retrieve detailed information about a specific CVE ID.
        """
        print("get_cve_details running from CVETool")
        url = f"{self.BASE_URL}?cveId={cve_id}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            vulns = data.get("vulnerabilities", [])
            if vulns:
                return vulns[0]
        return None

    def get_related_exploits(self, cve_id: str) -> List[Dict]:
        """
        Fetch related exploits for a given CVE ID.
        Placeholder using Exploit-DB search URL
        """
        print("get_related_exploits running from CVETool")
        return [{"cve": cve_id, "source": "Exploit-DB", "url": f"https://www.exploit-db.com/search?cve={cve_id}"}]

    def get_patch_info(self, cve_id: str) -> dict:
        """
        Retrieve patch or mitigation info for a CVE.
        Returns:
            {
                "patches": list of valid patch URLs,
                "message": informational message if no patches found
            }
        - Skips broken links and invalid URLs
        - Uses fallback keyword scan if no tagged patches
        """
        print("get_patch_info running from CVETool")
        details = self.get_cve_details(cve_id)
        if not details:
            return {"patches": [], "message": "CVE details not found."}

        cve_data = details.get("cve", {})
        references = cve_data.get("references", [])

        patches = []

        # 1️⃣ Collect references explicitly tagged as 'patch'
        for ref in references:
            tags = [t.lower() for t in ref.get("tags", [])]
            url = ref.get("url", "")
            if "broken link" in tags or not url.startswith("http"):
                continue
            if "patch" in tags:
                patches.append(url)

        # 2️⃣ Fallback: scan URLs for keywords if no tagged patches
        if not patches:
            keywords = ["patch", "update", "fix"]
            for ref in references:
                url = ref.get("url", "")
                tags = [t.lower() for t in ref.get("tags", [])]
                if "broken link" in tags or not url.startswith("http"):
                    continue
                if any(k in url.lower() for k in keywords):
                    patches.append(url)

        message = ""
        if not patches:
            message = "No valid patch URLs found. Please check the vendor’s official site for updates."

        return {"patches": patches, "message": message}
