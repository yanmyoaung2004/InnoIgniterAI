from mitreattack.stix20 import MitreAttackData

class SafeMitreTool:
    DOMAINS = ["enterprise-attack", "mobile-attack", "ics-attack"]

    def __init__(self):
        self.data = {}
        for domain in self.DOMAINS:
            filename = f"app/data/mitre/{domain}.json"
            try:
                self.data[domain] = MitreAttackData(filename)
                print(f"[+] Loaded {domain} dataset from {filename}")
            except FileNotFoundError:
                print(f"[!] File not found: {filename}")

    def _get_external_id(self, obj):
        print("get_external_id running from MITRETool")
        for ref in getattr(obj, "external_references", []):
            if "external_id" in ref:
                return ref["external_id"]
        return None

    def search_techniques(self, keyword: str):
        print("search_techniques running from MITRETool")
        results = []
        for domain, attack_data in self.data.items():
            for t in attack_data.get_objects_by_type("attack-pattern"):
                t_name = getattr(t, "name", "")
                t_desc = getattr(t, "description", "")
                t_id = self._get_external_id(t)
                if not t_id:
                    continue
                if keyword.lower() in t_name.lower() or keyword.lower() in t_desc.lower():
                    results.append({
                        "domain": domain,
                        "tech_id": t_id,
                        "technique": t_name,
                        "description": t_desc
                    })
        return results

    def get_technique_by_id(self, attack_id: str):
        print("get_technique_by_id running from MITRETool")
        for domain, attack_data in self.data.items():
            t_obj = attack_data.get_object_by_attack_id(attack_id, "attack-pattern")
            if t_obj:
                return {
                    "domain": domain,
                    "tech_id": attack_id,
                    "technique": getattr(t_obj, "name", ""),
                    "description": getattr(t_obj, "description", "")
                }
        return None

    def get_mitigations_for_technique(self, attack_id: str):
        print("get_mitigations_for_technique running from MITRETool")
        for domain, attack_data in self.data.items():
            t_obj = attack_data.get_object_by_attack_id(attack_id, "attack-pattern")
            if not t_obj:
                continue

            mitigations = []
            for r in attack_data.get_objects_by_type("relationship"):
                if getattr(r, "relationship_type", "") == "mitigates" and getattr(r, "target_ref", "") == t_obj.id:
                    mit_obj = attack_data.get_object_by_stix_id(r.source_ref)
                    if mit_obj:
                        mitigations.append({
                            "mitigation_id": self._get_external_id(mit_obj),
                            "name": getattr(mit_obj, "name", ""),
                            "description": getattr(mit_obj, "description", "")
                        })
            return {
                "domain": domain,
                "technique_id": attack_id,
                "technique": getattr(t_obj, "name", ""),
                "mitigations": mitigations
            }
        return None

    def get_mitigations_by_keyword(self, keyword: str):
        print("get_mitigations_by_keyword running from MITRETool")
        results = []
        for domain, attack_data in self.data.items():
            for t in attack_data.get_objects_by_type("attack-pattern"):
                t_name = getattr(t, "name", "")
                t_desc = getattr(t, "description", "")
                t_id = self._get_external_id(t)
                if not t_id:
                    continue
                if keyword.lower() in t_name.lower() or keyword.lower() in t_desc.lower():
                    mitigations = []
                    for r in attack_data.get_objects_by_type("relationship"):
                        if getattr(r, "relationship_type", "") == "mitigates" and getattr(r, "target_ref", "") == t.id:
                            mit_obj = attack_data.get_object_by_stix_id(r.source_ref)
                            if mit_obj:
                                mitigations.append({
                                    "mitigation_id": self._get_external_id(mit_obj),
                                    "name": getattr(mit_obj, "name", ""),
                                    "description": getattr(mit_obj, "description", "")
                                })
                    results.append({
                        "domain": domain,
                        "technique_id": t_id,
                        "technique": t_name,
                        "mitigations": mitigations
                    })
        return results

    def get_techniques_for_mitigation(self, mitigation_id: str):
        print("get_techniques_for_mitigation running from MITRETool")      
        results = []
        for domain, attack_data in self.data.items():
            mit_obj = attack_data.get_object_by_attack_id(mitigation_id, "course-of-action")
            if not mit_obj:
                continue
            for r in attack_data.get_objects_by_type("relationship"):
                if getattr(r, "relationship_type", "") == "mitigates" and getattr(r, "source_ref", "") == mit_obj.id:
                    tech_obj = attack_data.get_object_by_stix_id(r.target_ref)
                    if tech_obj:
                        results.append({
                            "domain": domain,
                            "technique_id": self._get_external_id(tech_obj),
                            "technique": getattr(tech_obj, "name", ""),
                            "description": getattr(tech_obj, "description", "")
                        })
        return results

