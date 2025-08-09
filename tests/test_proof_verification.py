#!/usr/bin/env python3
"""
Test script to verify the proof verification functionality in badge lookup
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
from modules.constants import CRYPTO_PK

def test_proof_verification():
    """Test the proof verification functionality for badge lookup"""
    print("🧪 Testing Proof Verification Integration")
    print("=" * 50)
    
    # Check if crypto utilities are available
    print(f"\n1. Checking crypto utilities availability...")
    try:
        from modules.crypto_utils import (
            create_complete_signed_credential, 
            generate_key_id, 
            verify_credential_proof,
            verify_signature,
            create_proof_hash
        )
        print("   ✅ All crypto utilities imported successfully")
        crypto_available = True
    except ImportError as e:
        print(f"   ❌ Failed to import crypto utilities: {e}")
        crypto_available = False
        return False
    
    # Test verification functions with sample data
    print(f"\n2. Testing verification functions...")
    if crypto_available and CRYPTO_PK is not None:
        try:
            # Create a sample signed credential for testing
            badge_guid = "test-12345678-1234-5678-1234-567812345678"
            issuer = {
                "id": "https://test.edu/issuers/565049",
                "type": "Profile",
                "name": "Test University"
            }
            achievement = {
                "id": f"{issuer['id']}/achievements/{badge_guid}",
                "type": ["Achievement"],
                "name": "Test Verification Badge",
                "description": "A test badge for verifying proof verification functionality",
                "criteria": {"narrative": "Complete the verification test"}
            }
            
            # Generate signed credential
            issuer_key_id = generate_key_id(issuer["id"], 1)
            current_time = datetime.utcnow().isoformat() + "Z"
            
            signed_credential = create_complete_signed_credential(
                credential_id=f"urn:uuid:{badge_guid}",
                subject_id="did:example:test-verification",
                issuer=issuer,
                achievement=achievement,
                issuer_key_id=issuer_key_id,
                valid_from=current_time,
                name="Test Verification Badge for Testing",
                description="Badge awarded for testing verification"
            )
            
            print("   ✅ Sample signed credential created")
            
            # Test proof verification
            is_valid, status_message = verify_credential_proof(signed_credential)
            
            if is_valid:
                print("   ✅ Proof verification successful")
                print(f"      Status: {status_message}")
            else:
                print("   ❌ Proof verification failed")
                print(f"      Status: {status_message}")
                return False
                
            # Test with tampered credential
            print(f"\n3. Testing tampered credential detection...")
            tampered_credential = signed_credential.copy()
            tampered_credential["credentialSubject"]["achievement"]["name"] = "TAMPERED ACHIEVEMENT"
            
            is_valid_tampered, status_tampered = verify_credential_proof(tampered_credential)
            
            if not is_valid_tampered:
                print("   ✅ Tampered credential correctly detected as invalid")
                print(f"      Status: {status_tampered}")
            else:
                print("   ❌ Tampered credential incorrectly validated as valid")
                return False
            
            # Test with missing proof
            print(f"\n4. Testing credential without proof...")
            no_proof_credential = signed_credential.copy()
            del no_proof_credential["proof"]
            
            is_valid_no_proof, status_no_proof = verify_credential_proof(no_proof_credential)
            
            if not is_valid_no_proof and "No proof section" in status_no_proof:
                print("   ✅ Credential without proof correctly detected")
                print(f"      Status: {status_no_proof}")
            else:
                print("   ❌ Credential without proof not properly handled")
                return False
            
            # Test with missing verification method
            print(f"\n5. Testing credential without verification method...")
            no_vm_credential = signed_credential.copy()
            del no_vm_credential["verificationMethod"]
            
            is_valid_no_vm, status_no_vm = verify_credential_proof(no_vm_credential)
            
            if not is_valid_no_vm and "No verification method" in status_no_vm:
                print("   ✅ Credential without verification method correctly detected")
                print(f"      Status: {status_no_vm}")
            else:
                print("   ❌ Credential without verification method not properly handled")
                return False
            
        except Exception as e:
            print(f"   ❌ Error during verification testing: {e}")
            return False
    else:
        print("   ⏭️ Skipping verification test (CRYPTO_PK not available or crypto not available)")
    
    print(f"\n🎉 PROOF VERIFICATION TEST COMPLETED!")
    print(f"\n📊 Summary:")
    print(f"   - Crypto utilities available: {'✅' if crypto_available else '❌'}")
    print(f"   - CRYPTO_PK available: {'✅' if CRYPTO_PK is not None else '❌'}")
    print(f"   - Proof verification functional: {'✅' if (crypto_available and CRYPTO_PK is not None) else '❌'}")
    
    if crypto_available and CRYPTO_PK is not None:
        print(f"\n✅ Badge proof verification is ENABLED")
        print(f"   - Badge lookup will verify cryptographic proofs")
        print(f"   - Tampered credentials will be detected")
        print(f"   - Detailed verification status will be displayed")
        print(f"   - Unsigned badges will be identified")
    else:
        print(f"\n📝 Badge proof verification is LIMITED")
        if not crypto_available:
            print(f"   - Crypto utilities not available")
        if CRYPTO_PK is None:
            print(f"   - CRYPTO_PK environment variable not set")
        print(f"   - Badge lookup will show limited verification status")
    
    return True

if __name__ == "__main__":
    success = test_proof_verification()
    sys.exit(0 if success else 1)