/*** Author :Vibhav Gogate
The University of Texas at Dallas
 *****/

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;

public class KMeans {
	public static void main(String[] args) {
		if (args.length < 3) {
			System.out
					.println("Usage: Kmeans <input-image> <k> <output-image>");
			return;
		}
		try {
			BufferedImage originalImage = ImageIO.read(new File(args[0]));
			int k = Integer.parseInt(args[1]);
			BufferedImage kmeansJpg = kmeans_helper(originalImage, k);
			ImageIO.write(kmeansJpg, "png", new File(args[2]));

		} catch (IOException e) {
			System.out.println(e.getMessage());
		}
	}

	private static BufferedImage kmeans_helper(BufferedImage originalImage,
			int k) {
		int w = originalImage.getWidth();
		int h = originalImage.getHeight();
		BufferedImage kmeansImage = new BufferedImage(w, h,
				originalImage.getType());
		Graphics2D g = kmeansImage.createGraphics();
		g.drawImage(originalImage, 0, 0, w, h, null);
		// Read rgb values from the image
		int[] rgb = new int[w * h];
		int count = 0;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				rgb[count++] = kmeansImage.getRGB(i, j);
			}
		}
		// Call kmeans algorithm: update the rgb values
		kmeans(rgb, k);

		// Write the new rgb values to the image
		count = 0;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				kmeansImage.setRGB(i, j, rgb[count++]);
			}
		}
		return kmeansImage;
	}

	// Your k-means code goes here
	// Update the array rgb by assigning each entry in the rgb array to its
	// cluster center
	private static void kmeans(int[] rgb, int k) {
		int[] prevClusterCentre = new int[k];
		int[] clusterCentre = new int[k];
		int[] clusters = new int[rgb.length];
		int closestCentre = 0;
		double maxDistance = Double.MAX_VALUE;
		double distance = 0;
		for (int i = 0; i < clusterCentre.length; i++) {
			Random random = new Random();
			clusterCentre[i] = rgb[random.nextInt(rgb.length)];
		}
		do {
			double[] pixelCount = new double[k];
			int[] r = new int[k];
			int[] g = new int[k];
			int[] b = new int[k];
			int[] alpha = new int[k];
			for (int i = 0; i < clusterCentre.length; i++)
				prevClusterCentre[i] = clusterCentre[i];
			for (int i = 0; i < rgb.length; i++) {
				maxDistance = Double.MAX_VALUE;
				for (int j = 0; j < clusterCentre.length; j++) {
					distance = getEuclideanDistance(rgb[i], clusterCentre[j]);
					if (distance < maxDistance) {
						maxDistance = distance;
						closestCentre = j;
					}
				}
				clusters[i] = closestCentre;
				pixelCount[closestCentre]++;
				alpha[closestCentre] += ((rgb[i] & 0xFF000000) >> 24);
				r[closestCentre] += ((rgb[i] & 0xFF0000) >> 16);
				g[closestCentre] += ((rgb[i] & 0xFF00) >> 8);
				b[closestCentre] += (rgb[i] & 0xFF);
			}
			for (int i = 0; i < clusterCentre.length; i++) {
				int A = (int) (alpha[i] / pixelCount[i]);
				int R = (int) (r[i] / pixelCount[i]);
				int G = (int) (g[i] / pixelCount[i]);
				int B = (int) (b[i] / pixelCount[i]);
				clusterCentre[i] = ((A & 0xFF) << 24) | ((R & 0xFF) << 16)
						| ((G & 0xFF) << 8) | (B & 0xFF);
			}
		} while (!checkConvergence(prevClusterCentre, clusterCentre));
		for (int i = 0; i < rgb.length; i++)
			rgb[i] = clusterCentre[clusters[i]];
	}
	private static double getEuclideanDistance(int pixel1, int pixel2) {
		int A = ((pixel1 & 0xFF000000) >> 24) - ((pixel2 & 0xFF000000) >> 24);
		int R = ((pixel1 & 0xFF0000) >> 16) - ((pixel2 & 0xFF0000) >> 16);
		int G = ((pixel1 & 0xFF00) >> 8) - ((pixel2 & 0xFF00) >> 8);
		int B = (pixel1 & 0xFF) - (pixel2 & 0xFF);
		return Math.sqrt(A * A + R * R + G * G + B * B);
	}
	private static boolean checkConvergence(int[] list1, int[] list2) {
		for (int i = 0; i < list1.length; i++)
			if (list1[i] != list2[i])
				return false;
		return true;
	}
}