' Lanzador de doble clic de la Consola de operador: abre la VENTANA NATIVA (pywebview) sin consola.
' Usa el venv del repo y los artefactos (modelos + datos ERA5/procesados) en su sitio -> no empaqueta
' nada (los datos del proyecto son varios GB; ver webapp/README.md). Cierra la ventana para salir.
Dim sh, fso, base, root, pyw, script
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
base = fso.GetParentFolderName(WScript.ScriptFullName)   ' ...\webapp
root = fso.GetParentFolderName(base)                     ' raiz del repo
pyw = fso.BuildPath(root, ".venv\Scripts\pythonw.exe")
script = fso.BuildPath(base, "desktop.py")
sh.CurrentDirectory = root
If Not fso.FileExists(pyw) Then
  MsgBox "No encuentro el entorno virtual: " & pyw & vbCrLf & _
         "Crea el venv e instala:  pip install -e "".[webapp]""", 48, "Consola de operador"
ElseIf Not fso.FileExists(script) Then
  MsgBox "No encuentro " & script, 48, "Consola de operador"
Else
  sh.Run """" & pyw & """ """ & script & """", 0, False
End If
