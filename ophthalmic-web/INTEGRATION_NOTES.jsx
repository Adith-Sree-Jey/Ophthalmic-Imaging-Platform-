/*
Frontend integration checklist for the MS SQL history workflow.

Completed wiring in this repo:

1. App routes
   - /dashboard
   - /history

2. Navbar links
   - Dashboard
   - Patient History

3. Upload flow
   - EyeSideSelector is rendered in UploadPanel
   - /classify sends eye_side with OD/OS

4. API helpers
   - fetchAllPatients()
   - fetchPatientHistory(mriNumber)
   - fetchLongitudinal(mriNumber, eyeSide)

5. History page
   - quick pick list of patients
   - MRI search
   - per-eye visit cards
   - SVG progression chart

Token note:
The app stores both localStorage keys:
  - token
  - access_token

This keeps the older dashboard flow and the newer history helpers in sync.
*/
