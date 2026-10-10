package com.d1omni

import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import org.junit.Assert.assertEquals
import org.junit.Test
import org.w3c.dom.Element

/**
 * The app's own permissions (the merged manifest adds the libraries' on top): the microphone and the phone's
 * pictures only — READ_EXTERNAL_STORAGE up to Android 12 (API 32), READ_MEDIA_IMAGES from 13 — and no network.
 */
class D1ManifestTest {
  @Test
  fun permissions() {
    val doc = DocumentBuilderFactory.newInstance().apply { isNamespaceAware = true }.newDocumentBuilder()
      .parse(File("src/main/AndroidManifest.xml"))
    val android = "http://schemas.android.com/apk/res/android"
    val uses = doc.getElementsByTagName("uses-permission")
    val found = LinkedHashMap<String, String?>()
    for (index in 0 until uses.length) {
      val element = uses.item(index) as Element
      found[element.getAttributeNS(android, "name")] = element.getAttributeNS(android, "maxSdkVersion").ifEmpty { null }
    }
    assertEquals(
      linkedMapOf(
        "android.permission.RECORD_AUDIO" to null,
        "android.permission.READ_MEDIA_IMAGES" to null,
        "android.permission.READ_EXTERNAL_STORAGE" to "32",
      ),
      found,
    )
  }
}
