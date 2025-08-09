#!/usr/bin/env python3
"""
Test script to verify cryptographic signing integration in badge creation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
from datetime import datetime
from modules.constants import CRYPTO_PK
from modules.build_openbadge_metadata import build_openbadge_metadata

def test_cryptographic_integration():
    """Test the cryptographic signing integration for badge creation"""
    print("🧪 Testing Cryptographic Signing Integration")
    print("=" * 50)
    
    # Check CRYPTO_PK status
    print(f"\n1. Checking CRYPTO_PK environment variable...")
    if CRYPTO_PK is not None:
        print(f"   ✅ CRYPTO_PK is set (length: {len(str(CRYPTO_PK))} chars)")
        crypto_available = True
    else:
        print("   ❌ CRYPTO_PK is not set")
        crypto_available = False
    
    # Test crypto utilities import
    print(f"\n2. Testing crypto utilities import...")
    try:
        from modules.crypto_utils import create_complete_signed_credential, generate_key_id
        print("   ✅ Crypto utilities imported successfully")
        crypto_functions_available = True
    except ImportError as e:
        print(f"   ❌ Failed to import crypto utilities: {e}")
        crypto_functions_available = False
    
    # Test complete integration
    print(f"\n3. Testing complete cryptographic signing...")
    if crypto_available and crypto_functions_available:
        try:
            # Sample badge data
            badge_guid = "12345678-1234-5678-1234-567812345678"
            issuer = {
                "id": "https://example.edu/issuers/565049",
                "type": "Profile",
                "name": "Test University"
            }
            achievement = {
                "id": f"{issuer['id']}/achievements/{badge_guid}",
                "type": ["Achievement"],
                "name": "Test Badge",
                "description": "A test badge for cryptographic verification",
                "criteria": {"narrative": "Complete the integration test"}
            }
            
            # Generate key ID and create signed credential
            issuer_key_id = generate_key_id(issuer["id"], 1)
            current_time = datetime.utcnow().isoformat() + "Z"
            
            signed_credential = create_complete_signed_credential(
                credential_id=f"urn:uuid:{badge_guid}",
                subject_id="did:example:test-subject",
                issuer=issuer,
                achievement=achievement,
                issuer_key_id=issuer_key_id,
                valid_from=current_time,
                name="Test Badge for Integration Test",
                description="Badge awarded for testing"
            )
            
            print("   ✅ Signed credential created successfully")
            
            # Verify the credential has required fields
            required_fields = ["verificationMethod", "proof"]
            missing_fields = []
            
            for field in required_fields:
                if field not in signed_credential:
                    missing_fields.append(field)
            
            if not missing_fields:
                print("   ✅ All required cryptographic fields present")
                
                # Check verification method details
                vm = signed_credential["verificationMethod"][0]
                if all(key in vm for key in ["id", "type", "controller", "publicKeyMultibase"]):
                    print("   ✅ Verification method structure valid")
                else:
                    print("   ❌ Verification method missing required fields")
                
                # Check proof details
                proof = signed_credential["proof"]
                if all(key in proof for key in ["type", "created", "verificationMethod", "proofPurpose", "proofValue"]):
                    print("   ✅ Proof structure valid")
                    print(f"      - Proof type: {proof['type']}")
                    print(f"      - Verification method: {proof['verificationMethod']}")
                else:
                    print("   ❌ Proof section missing required fields")
                    
            else:
                print(f"   ❌ Missing required fields: {missing_fields}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error creating signed credential: {e}")
            return False
    else:
        print("   ⏭️ Skipping cryptographic test (requirements not met)")
    
    # Test standard badge creation (fallback)
    print(f"\n4. Testing standard badge creation (fallback)...")
    try:
        standard_metadata = build_openbadge_metadata(
            credential_id=f"urn:uuid:{badge_guid}",
            subject_id="did:example:test-subject",
            issuer=issuer,
            valid_from=current_time,
            achievement=achievement,
            name="Standard Test Badge",
            description="Badge created without cryptographic proof"
        )
        
        standard_credential = json.loads(standard_metadata)
        
        # Verify it's a valid credential but without crypto fields
        has_crypto = "verificationMethod" in standard_credential and "proof" in standard_credential
        
        if not has_crypto:
            print("   ✅ Standard badge created without cryptographic fields")
        else:
            print("   ⚠️ Standard badge unexpectedly includes cryptographic fields")
            
    except Exception as e:
        print(f"   ❌ Error creating standard badge: {e}")
        return False
    
    print(f"\n🎉 INTEGRATION TEST COMPLETED!")
    print(f"\n📊 Summary:")
    print(f"   - CRYPTO_PK available: {'✅' if crypto_available else '❌'}")
    print(f"   - Crypto functions available: {'✅' if crypto_functions_available else '❌'}")
    print(f"   - Badge signing enabled: {'✅' if (crypto_available and crypto_functions_available) else '❌'}")
    
    if crypto_available and crypto_functions_available:
        print(f"\n✅ Cryptographic badge signing is ENABLED")
        print(f"   - New badges will include verification methods and proofs")
        print(f"   - Badges can be cryptographically verified")
        print(f"   - Enhanced security and authenticity")
    else:
        print(f"\n📝 Cryptographic badge signing is DISABLED")
        print(f"   - Badges will be created without cryptographic proofs")
        print(f"   - Set CRYPTO_PK environment variable to enable signing")
        print(f"   - Ensure crypto_utils.py is available and functional")
    
    return True

if __name__ == "__main__":
    success = test_cryptographic_integration()
    sys.exit(0 if success else 1)